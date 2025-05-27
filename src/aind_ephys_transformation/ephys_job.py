"""Module to handle ephys data compression"""

import logging
import os
import platform
import shutil
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal, Optional, Union

import numpy as np
from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)
from numcodecs.abc import Codec
from pydantic import Field

import probeinterface as pi
import spikeinterface as si
from spikeinterface import extractors as se
import spikeinterface.preprocessing as spre

from aind_ephys_transformation.models import (
    CompressorName,
    ReaderName,
    RecordingBlockPrefixes,
)


class EphysJobSettings(BasicJobSettings):
    """EphysCompressionJob settings."""

    # reader settings
    reader_name: ReaderName = Field(
        default=ReaderName.OPENEPHYS, description="Name of reader to use."
    )
    # Clip settings
    clip_n_frames: int = Field(
        default=100,
        description="Number of frames to clip the data.",
        title="Clip N Frames",
    )
    # Check timestamps alignment
    check_timestamps: bool = Field(
        default=True,
        description="Check if timestamps are aligned and raise an error if "
        "they are not.",
        title="Check Timestamps",
    )
    # Compress settings
    random_seed: Optional[int] = 0
    compress_write_output_format: Literal["zarr"] = Field(
        default="zarr",
        description=(
            "Output format for compression. Currently, only zarr supported."
        ),
        title="Write Output Format",
    )
    compress_max_windows_filename_len: int = Field(
        default=150,
        description=(
            "Windows OS max filename length is 256. The zarr write will "
            "raise an error if it detects that the destination directory has "
            "a long name."
        ),
        title="Compress Max Windows Filename Len",
    )
    compressor_name: CompressorName = Field(
        default=CompressorName.WAVPACK,
        description="Type of compressor to use.",
        title="Compressor Name.",
    )
    # It will be safer if these kwargs fields were objects with known schemas
    compressor_kwargs: dict = Field(
        default={"level": 3},
        description="Arguments to be used for the compressor.",
        title="Compressor Kwargs",
    )
    compress_job_save_kwargs: dict = Field(
        default={"n_jobs": -1},  # -1 to use all available cpu cores.
        description="Arguments for recording save method.",
        title="Compress Job Save Kwargs",
    )
    compress_chunk_duration: str = Field(
        default="1s",
        description="Duration to be used for chunks.",
        title="Compress Chunk Duration",
    )

    # Scale settings
    scale_num_chunks_per_segment: int = Field(
        default=100,
        description="Num of chunks per segment to scale.",
        title="Scale Num Chunks Per Segment",
    )
    scale_chunk_size: int = Field(
        default=10000,
        description="Chunk size to scale.",
        title="Scale Chunk Size",
    )


class EphysCompressionJob(GenericEtl[EphysJobSettings]):
    """Main class to handle ephys data compression"""

    def _get_read_blocks(self) -> Iterator[dict]:
        """
        Uses SpikeInterface to extract read blocks from the input source.

        Returns:
        Iterator[dict]
            A generator of read blocks. A single read_block is dict with keys:
            ('recording', 'experiment_name', 'stream_name')

        """
        if self.job_settings.reader_name == ReaderName.CHRONIC:
            dataset_folder = Path(self.job_settings.input_source)
            onix_folders = [p for p in dataset_folder.iterdir() if p.is_dir() and "OnixEphys" in p.name]
            assert len(onix_folders) == 1
            onix_folder = onix_folders[0]

            stream_name = "AmplifierData"
            amplifier_datasets = [p for p in onix_folder.iterdir() if stream_name in p.name and p.suffix == ".bin"]

            probe_json = dataset_folder / "probe.json"
            binary_info_json = dataset_folder / "binary_info.json"

            probe_group = pi.read_probeinterface(probe_json)

            with open(binary_info_json) as f:
                binary_info = json.load(f)

            adc_depth = binary_info.pop("adc_depth")

            # sort dates
            dates = [p.stem.split("_")[-1] for p in amplifier_datasets]

            for date in dates:
                amp_data = [p for p in amplifier_datasets if date in p.name][0]

                recording = si.read_binary(amp_data, **binary_info)
                recording = recording.set_probegroup(probe_group, group_mode="by_shank")

                # unsigned to signed
                rec = spre.unsigned_to_signed(recording, bit_depth=adc_depth)

                yield (
                    {
                        "recording": rec,
                        "experiment_name": date,
                        "stream_name": stream_name,
                    }
                )
        elif self.job_settings.reader_name == ReaderName.OPENEPHYS:
            nblocks = se.get_neo_num_blocks(
                self.job_settings.reader_name.value, self.job_settings.input_source
            )
            stream_names, _ = se.get_neo_streams(
                self.job_settings.reader_name.value, self.job_settings.input_source
            )
            # load first stream to map block_indices to experiment_names
            rec_test = se.read_openephys(
                self.job_settings.input_source,
                block_index=0,
                stream_name=stream_names[0],
            )
            record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
            experiments = rec_test.neo_reader.folder_structure[record_node][
                "experiments"
            ]
            exp_ids = list(experiments.keys())
            experiment_names = [
                experiments[exp_id]["name"] for exp_id in sorted(exp_ids)
            ]
            for block_index in range(nblocks):
                for stream_name in stream_names:
                    rec = se.read_openephys(
                        self.job_settings.input_source,
                        stream_name=stream_name,
                        block_index=block_index,
                        load_sync_timestamps=True,
                    )
                    yield (
                        {
                            "recording": rec,
                            "experiment_name": experiment_names[block_index],
                            "stream_name": stream_name,
                        }
                    )

    def _get_streams_to_clip(self) -> Iterator[dict]:
        """
        Returns
        -------
        Iterator[dict]
          A list of dicts with information on which streams to clip.
          The dictionary has keys ('data', 'relative_path_name', 'n_chan')
          The 'data' is a numpy.memmap object.

        """
        if self.job_settings.reader_name == ReaderName.CHRONIC:
            # Chronic data does not have .dat files, so we return an empty iterator
            amplifier_data_files = self.job_settings.input_source.glob("**/*AmplifierData*.bin")
            dataset_folder = Path(self.job_settings.input_source)
            binary_info_json = dataset_folder / "binary_info.json"
            with open(binary_info_json) as f:
                binary_info = json.load(f)
            dtype = binary_info.get("dtype", "uint16")
            order = "C" if binary_info.get("time_axis", 0) == 0 else "F"
            n_chan = binary_info.get("num_channels", 384)
            for amp_data_file in amplifier_data_files:
                data = np.memmap(
                    filename=str(amp_data_file),
                    dtype=dtype,
                    order=order,
                    mode="r",
                ).reshape(-1, n_chan)
                yield {
                    "data": data,
                    "relative_path_name": str(
                        amp_data_file.relative_to(self.job_settings.input_source)
                    ),
                    "n_chan": n_chan,
                }
        else:
            stream_names, _ = se.get_neo_streams(
                self.job_settings.reader_name.value, self.job_settings.input_source
            )
            for dat_file in self.job_settings.input_source.glob("**/*.dat"):
                oe_stream_name = dat_file.parent.name
                si_stream_name = [
                    stream_name
                    for stream_name in stream_names
                    if oe_stream_name in stream_name
                ][0]
                n_chan = se.read_openephys(
                    self.job_settings.input_source,
                    block_index=0,
                    stream_name=si_stream_name,
                ).get_num_channels()
                data = np.memmap(
                    filename=str(dat_file), dtype="int16", order="C", mode="r"
                ).reshape(-1, n_chan)
                yield {
                    "data": data,
                    "relative_path_name": str(
                        dat_file.relative_to(self.job_settings.input_source)
                    ),
                    "n_chan": n_chan,
                }

    def _check_timestamps_alignment(self) -> bool:
        """
        Check if timestamps have been aligned.
        This is done by checking if there are any original_timestamps.npy files
        in the openephys folder.

        Returns
        -------
        bool
          True if timestamps are aligned, False otherwise.
        """
        if self.job_settings.reader_name == ReaderName.CHRONIC:
            # Chronic data does not have timestamps, so we return True
            return True
        else:
            # OpenEphys data has timestamps, so we check for them
            return self._check_openephys_timestamps()

    def _check_openephys_timestamps(self) -> bool:
        openephys_folder = self.job_settings.input_source

        # Check if original_timestamps.npy files are present
        original_timestamps = [
            p for p in openephys_folder.glob("**/original_timestamps.npy")
        ]
        adjusted_timestamps_flag = [
            p for p in openephys_folder.glob("**/TIMESTAMPS_ADJUSTED.flag")
        ]

        if (
            len(original_timestamps) == 0
            and len(adjusted_timestamps_flag) == 0
        ):
            return False
        else:
            return True

    def _get_compressor(self) -> Codec:
        """
        Utility method to construct a compressor object.
        Returns
        -------
        Codec
          Either an instantiated numcodecs Codec object.

        """
        if self.job_settings.compressor_name == CompressorName.BLOSC:
            from numcodecs import Blosc
            return Blosc(**self.job_settings.compressor_kwargs)
        elif self.job_settings.compressor_name == CompressorName.WAVPACK:
            from wavpack_numcodecs import WavPack
            return WavPack(**self.job_settings.compressor_kwargs)
        else:
            # TODO: This is validated during the construction of JobSettings,
            #  so we can probably just remove this exception.
            raise Exception(
                f"Unknown compressor. Please select one of "
                f"{[c for c in CompressorName]}"
            )

    def _scale_read_blocks(
        self,
        read_blocks: Iterator[dict],
        random_seed: Optional[int] = None,
        num_chunks_per_segment: int = 100,
        chunk_size: int = 10000,
    ):
        """
        Scales read_blocks. A single read_block is dict with keys:
        ('recording', 'block_index', 'stream_name')
        Parameters
        ----------
        read_blocks : Iterator[dict]
          A single read_block is dict with keys:
          ('recording', 'block_index', 'stream_name')
        random_seed : Optional[int]
          Optional seed for correct_lsb method. Default is None.
        num_chunks_per_segment : int
          Default is 100
        chunk_size : int
          Default is 10000

        Returns
        -------
        Iterator[dict]
          An iterator over read_blocks. A single read_block is dict with keys:
          ('scale_recording', 'block_index', 'stream_name')
        """
        for read_block in read_blocks:
            # We don't need to scale the NI-DAQ recordings
            # TODO: Convert this to regex matching?
            if RecordingBlockPrefixes.nidaq.value in read_block["stream_name"]:
                rec_to_compress = read_block["recording"]
            else:
                correct_lsb_args = {
                    "num_chunks_per_segment": num_chunks_per_segment,
                    "chunk_size": chunk_size,
                }
                if random_seed is not None:
                    correct_lsb_args["seed"] = random_seed
                rec_to_compress = spre.correct_lsb(
                    read_block["recording"], **correct_lsb_args
                )
            yield (
                {
                    "scaled_recording": rec_to_compress,
                    "experiment_name": read_block["experiment_name"],
                    "stream_name": read_block["stream_name"],
                }
            )

    def _copy_and_clip_data(
        self,
        dst_dir: Path,
        stream_gen: Iterator[dict],
    ) -> None:
        """
        Copies the raw data to a new directory with the .dat files clipped to
        just a small number of frames. This allows someone to still use the
        spikeinterface api on the clipped data set.
        Parameters
        ----------
        dst_dir : Path
          Desired location for clipped data set
        stream_gen : Iterator[dict]
          An Iterator where each item is a dictionary with shape,
          ('data': memmap(dat file),
          'relative_path_name': path name of raw data to new dir correctly
          'n_chan': number of channels)
        Returns
        -------
        None
          None. Copies clipped *.dat files to a new directory.

        """

        # first: copy everything except .dat files
        if self.job_settings.reader_name == ReaderName.CHRONIC:
            # Chronic data does not have .dat files, so we copy everything
            # except the .dat files
            patterns_to_ignore = ["*AmplifierData*.bin"]
        elif self.job_settings.reader_name == ReaderName.OPENEPHYS:
            patterns_to_ignore = ["*.dat"]
        shutil.copytree(
            self.job_settings.input_source,
            dst_dir,
            ignore=shutil.ignore_patterns(*patterns_to_ignore),
        )
        # second: copy clipped dat files
        for stream in stream_gen:
            data = stream["data"]
            rel_path_name = stream["relative_path_name"]
            n_chan = stream["n_chan"]
            dst_raw_file = dst_dir / rel_path_name
            dst_data = np.memmap(
                filename=dst_raw_file,
                dtype="int16",
                shape=(self.job_settings.clip_n_frames, n_chan),
                order="C",
                mode="w+",
            )
            dst_data[:] = data[: self.job_settings.clip_n_frames]

    def _compress_and_write_block(
        self,
        read_blocks: Iterator[dict],
        compressor: Codec,
        output_dir: Path,
        job_kwargs: dict,
        max_windows_filename_len: int,
        output_format: str = "zarr",
    ) -> None:
        """
        Compress and write read_blocks.

        Parameters
        ----------
        read_blocks : Iterator[dict]
          Either [{'recording', 'block_index', 'stream_name'},...] or
          [{'scale_recording', 'block_index', 'stream_name'},...]
        compressor : Codec
        output_dir : Path
          Output directory to write compressed data
        job_kwargs : dict
          Recording save job kwargs
        max_windows_filename_len : int
          Windows OS has a maximum filename length. If the root directory is
          deeply nested, the zarr filenames might be too long for Windows.
          This is added as an arg, so we can raise better error messages.
        output_format : str
          Default is zarr

        Returns
        -------
        None
          Writes data to a folder.

        """
        if job_kwargs["n_jobs"] == -1:
            job_kwargs["n_jobs"] = os.cpu_count()

        for read_block in read_blocks:
            if "recording" in read_block:
                rec = read_block["recording"]
            else:
                rec = read_block["scaled_recording"]
            experiment_name = read_block["experiment_name"]
            stream_name = read_block["stream_name"]
            zarr_path = output_dir / f"{experiment_name}_{stream_name}.zarr"
            if (
                platform.system() == "Windows"
                and len(str(zarr_path)) > max_windows_filename_len
            ):
                raise Exception(
                    f"File name for zarr path is too long "
                    f"({len(str(zarr_path))})"
                    f" and might lead to errors. Use a shorter destination "
                    f"path."
                )
            # compression for times is disabled
            _ = rec.save(
                format=output_format,
                folder=zarr_path,
                compressor=compressor,
                compressor_by_dataset=dict(times=None),
                **job_kwargs,
            )

    def _compress_raw_data(self) -> None:
        """Compresses ephys data"""
        # Check if timestamps are aligned
        logging.info("Checking timestamps alignment.")
        timestamps_ok = self._check_timestamps_alignment()
        if not timestamps_ok:
            if self.job_settings.check_timestamps:
                raise Exception(
                    "Timestamps are not aligned. Please align timestamps "
                    "using aind-ephys-rig-qc before compressing the data."
                )
            else:
                logging.warning(
                    "Timestamps are not aligned, but timestamps check is "
                    "disabled. Proceeding with compression."
                )

        # Clip the data
        logging.info("Clipping source data. This may take a minute.")
        clipped_data_path = (
            self.job_settings.output_directory / "ecephys_clipped"
        )
        streams_to_clip = self._get_streams_to_clip()
        self._copy_and_clip_data(
            dst_dir=clipped_data_path,
            stream_gen=streams_to_clip,
        )

        logging.info("Finished copying and clipping source data.")

        # Compress the data
        logging.info("Compressing source data.")
        compressed_data_path = (
            self.job_settings.output_directory / "ecephys_compressed"
        )
        read_blocks = self._get_read_blocks()
        compressor = self._get_compressor()
        if self.job_settings.reader_name == ReaderName.CHRONIC:
            # No need to scale the Chronic Onix recordings
            scaled_read_blocks = read_blocks
        elif self.job_settings.reader_name == ReaderName.OPENEPHYS:
            # Scale the OpenEphys recordings
            scaled_read_blocks = self._scale_read_blocks(
                read_blocks=read_blocks,
                random_seed=self.job_settings.random_seed,
                num_chunks_per_segment=(
                    self.job_settings.scale_num_chunks_per_segment
                ),
                chunk_size=self.job_settings.scale_chunk_size,
            )
        self._compress_and_write_block(
            read_blocks=scaled_read_blocks,
            compressor=compressor,
            max_windows_filename_len=(
                self.job_settings.compress_max_windows_filename_len
            ),
            output_dir=compressed_data_path,
            output_format=self.job_settings.compress_write_output_format,
            job_kwargs=self.job_settings.compress_job_save_kwargs,
        )
        logging.info("Finished compressing source data.")

        return None

    def run_job(self) -> JobResponse:
        """
        Main public method to run the compression job
        Returns
        -------
        JobResponse
          Information about the job that can be used for metadata downstream.

        """
        job_start_time = datetime.now()
        self._compress_raw_data()
        job_end_time = datetime.now()
        return JobResponse(
            status_code=200,
            message=f"Job finished in: {job_end_time-job_start_time}",
            data=None,
        )


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    parser = get_parser()
    cli_args = parser.parse_args(sys_args)
    if cli_args.job_settings is not None:
        job_settings = EphysJobSettings.model_validate_json(
            cli_args.job_settings
        )
    elif cli_args.config_file is not None:
        job_settings = EphysJobSettings.from_config_file(cli_args.config_file)
    else:
        # Construct settings from env vars
        job_settings = EphysJobSettings()
    job = EphysCompressionJob(job_settings=job_settings)
    job_response = job.run_job()
    logging.info(job_response.model_dump_json())
