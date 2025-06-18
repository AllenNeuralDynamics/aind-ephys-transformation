"""Module to handle ephys data compression"""

import logging
import os
import platform
import shutil
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal, Optional, List
from pydantic import model_validator, field_validator

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
from aind_ephys_transformation.utils import sync_dir_to_s3, copy_file_to_s3


def extract_datetime(filename):
    """Extract datetime from filename."""
    # Extract datetime from filename in format YYYY-MM-DDThh-mm-ss
    date_str = filename.stem.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S")


def extract_datetime(filename):
    """Extract datetime from filename."""
    # Extract datetime from filename in format YYYY-MM-DDThh-mm-ss
    date_str = filename.stem.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S")


class EphysJobSettings(BasicJobSettings):
    """EphysCompressionJob settings."""

    s3_location: Optional[str] = Field(
        default=None,
        description=(
            "S3 location to upload the compressed data. "
            "If provided, the data will be uploaded to S3 and not saved "
            "to the output directory."
        ),
        title="S3 Location",
    )
    # reader settings
    reader_name: ReaderName = Field(
        default=ReaderName.OPENEPHYS,
        description="Name of reader to use.",
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
    chronic_chunks_to_compress: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of chunks to compress (chronic). "
            "Each element should be the date in the chunk file name. "
            "If None, all chunks in the input source folder will "
            "be compressed."
        ),
        title="Chunks to Compress",
    )
    chronic_start_flag: bool = Field(
        default=False,
        description=(
            "If True, it signals that the data to compress is the first "
            "chunk(s) of a chronic recording. This will be used to set "
            "additional attirbutes to the zarr groups."
        ),
        title="Chronic Start Flag",
    )
    check_chronic_consecutive_hours: bool = Field(
        default=True,
        description=(
            "Check if recordings are taken at 1-hour intervals. "
            "If True, raises error; if False, logs warning."
        ),
        title="Check Consecutive Hours",
    )
    hour_interval_tolerance: float = Field(
        default=0.05,
        description=(
            "Tolerance (as fraction) for time interval between "
            "recordings. Default 0.05 (5%)"
        ),
        title="Hour Interval Tolerance",
        ge=0.0,
        le=1.0,
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

    @model_validator(mode="after")
    def validate_chronic_chunks_to_compress(self):
        """Fills chronic_chunks_to_compress fro chronic reader if None."""
        if self.reader_name == ReaderName.CHRONIC:
            if self.chronic_chunks_to_compress is None:
                dates = [
                    p.stem.split("_")[-1]
                    for p in Path(self.input_source).glob(
                        "**/OnixEphys_AmplifierData_*.bin"
                    )
                ]
                self.chronic_chunks_to_compress = dates
        return self

<<<<<<< HEAD
    @field_validator("s3_location")
    @classmethod
    def validate_s3_location(cls, v: Optional[str]) -> Optional[str]:
        """Validates s3_location."""
        if v is not None:
            if not v.startswith("s3://"):
                raise ValueError(
                    "S3 location must start with 's3://'."
                )
            # remuve trailing slash if present
            if v.endswith("/"):
                v = v[:-1]
        return v

=======
>>>>>>> 5fb362a2bc226c94f74fdff5768fed49f76f3fe5

class EphysCompressionJob(GenericEtl[EphysJobSettings]):
    """Main class to handle ephys data compression"""

    def _get_read_blocks(self) -> Iterator[dict]:  # noqa: C901
        """
        Uses SpikeInterface to extract read blocks from the input source.

        Returns:
        Iterator[dict]
            A generator of read blocks. A single read_block is dict with keys:
            ('recording', 'experiment_name', 'stream_name')

        """
        if self.job_settings.reader_name == ReaderName.CHRONIC:
            dataset_folder = Path(self.job_settings.input_source)
            onix_folders = [
                p
                for p in dataset_folder.iterdir()
                if p.is_dir() and "OnixEphys" in p.name
            ]
            assert len(onix_folders) == 1
            onix_folder = onix_folders[0]

            stream_name = "AmplifierData"
            amplifier_datasets = [
                p
                for p in onix_folder.iterdir()
                if stream_name in p.name and p.suffix == ".bin"
            ]

            # filter the datasets based on the chronic_chunks_to_compress
            amplifier_datasets_to_compress = []
            for chunk_date in self.job_settings.chronic_chunks_to_compress:
                matching_datasets = [
                    p for p in amplifier_datasets if chunk_date in p.name
                ]
                if len(matching_datasets) == 0:
                    raise ValueError(
                        f"Date {chunk_date} not found in {stream_name} "
                        "datasets."
                    )
                elif len(matching_datasets) > 1:
                    raise ValueError(
                        f"Multiple datasets found for date {chunk_date} "
                        f"in {stream_name} datasets: {matching_datasets}. "
                        "Please specify a more specific chunk."
                    )
                amplifier_datasets_to_compress.append(matching_datasets[0])

            # Sort by date
            amplifier_datasets_to_compress = sorted(
                amplifier_datasets_to_compress,
                key=lambda x: extract_datetime(x),
            )

            # Check if recordings are consecutive with tolerance
            target_interval = 1.0  # 1 hour
            tolerance = self.job_settings.hour_interval_tolerance
            allowed_min = target_interval * (1 - tolerance)
            allowed_max = target_interval * (1 + tolerance)

            for i in range(len(amplifier_datasets_to_compress) - 1):
                curr_dt = extract_datetime(amplifier_datasets_to_compress[i])
                next_dt = extract_datetime(
                    amplifier_datasets_to_compress[i + 1]
                )
                time_diff = next_dt - curr_dt
                hour_diff = time_diff.total_seconds() / 3600

                if not (allowed_min <= hour_diff <= allowed_max):
                    message = (
                        f"Time interval outside tolerance range "
                        f"({tolerance*100}%) found between "
                        f"{curr_dt.strftime('%Y-%m-%dT%H-%M-%S')} and "
                        f"{next_dt.strftime('%Y-%m-%dT%H-%M-%S')}. "
                        f"Time difference: {hour_diff:.2f} hours "
                        f"(allowed range: {allowed_min:.2f} to "
                        f"{allowed_max:.2f} hours)"
                    )
                    if self.job_settings.check_chronic_consecutive_hours:
                        raise ValueError(message)
                    else:
                        logging.warning(message)

            # Parse probe and binary info
            probe_json = dataset_folder / "probe.json"
            binary_info_json = dataset_folder / "binary_info.json"
            probe_group = pi.read_probeinterface(probe_json)
            with open(binary_info_json) as f:
                binary_info = json.load(f)
            adc_depth = binary_info.pop("adc_depth")

            recording_list = []
            if self.job_settings.chronic_start_flag:
                sample_index_from_session_start = 0
            else:
                # If not chronic start flag, we need to parse all previous
                # clock bin files to get the cumulative start frame
                first_chunk_to_compress = (
                    self.job_settings.chronic_chunks_to_compress[0]
                )
                first_date_to_compress = datetime.strptime(
                    first_chunk_to_compress, "%Y-%m-%dT%H-%M-%S"
                )
                all_previous_clock_files = [
                    p
                    for p in dataset_folder.glob("**/OnixEphys_Clock_*")
                    if extract_datetime(p) < first_date_to_compress
                ]
                sorted_clock_files = sorted(
                    all_previous_clock_files,
                    key=lambda x: extract_datetime(x),
                )
                sample_index_from_session_start = 0
                for clock_file in sorted_clock_files:
                    clock_data = np.memmap(
                        filename=clock_file, dtype="uint64", mode="r"
                    )
                    sample_index_from_session_start += len(clock_data)
            logging.info(
                f"Sample index from session start: "
                f"{sample_index_from_session_start}"
            )

            start_end_frames = {}
            cumulative_start_frame = sample_index_from_session_start
            for amplifier_dataset in amplifier_datasets_to_compress:
                recording = si.read_binary(amplifier_dataset, **binary_info)

                # unsigned to signed
                recording = spre.unsigned_to_signed(
                    recording, bit_depth=adc_depth
                )

                # keep track start and end frames for each chunk
                start_end_frames[amplifier_dataset.name] = (
                    cumulative_start_frame,
                    cumulative_start_frame + recording.get_num_frames(),
                )
                # update cumulative start frame
                cumulative_start_frame += recording.get_num_frames()

                recording_list.append(recording)

            # concatenate recordings
            recording_concatenated = si.concatenate_recordings(recording_list)
            # set probe
            recording_concatenated = recording_concatenated.set_probegroup(
                probe_group, group_mode="by_shank"
            )
            # annotate recording with start and end frames
            recording_concatenated.annotate(start_end_frames=start_end_frames)
            recording_concatenated.annotate(
                sample_index_from_session_start=sample_index_from_session_start
            )

            yield (
                {
                    "recording": recording_concatenated,
                    "experiment_name": "experiment1",
                    "stream_name": stream_name,
                }
            )
        elif self.job_settings.reader_name == ReaderName.OPENEPHYS:
            nblocks = se.get_neo_num_blocks(
                self.job_settings.reader_name.value,
                self.job_settings.input_source,
            )
            stream_names, _ = se.get_neo_streams(
                self.job_settings.reader_name.value,
                self.job_settings.input_source,
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
            # For chronic, we don't have .dat files to clip, so we
            # return an empty iterator
            return iter([])
        else:
            stream_names, _ = se.get_neo_streams(
                self.job_settings.reader_name.value,
                self.job_settings.input_source,
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
                    filename=str(dat_file),
                    dtype="int16",
                    order="C",
                    mode="r",
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
        """
        Check if timestamps have been aligned in OpenEphys data.
        """
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

    def _copy_and_clip_data(  # noqa: C901
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
            # For chronic data we copy all .bin files except the
            # AmplifierData.bin files, which are compressed
            files_to_copy = []
            for date in self.job_settings.chronic_chunks_to_compress:
                files_to_copy.extend(
                    [
                        p
                        for p in self.job_settings.input_source.glob(
                            f"**/*{date}*"
                        )
                        if "AmplifierData" not in p.name and p.is_file()
                    ]
                )
            for f in files_to_copy:
                dst_file_path = dst_dir / f.relative_to(
                    self.job_settings.input_source
                )
                if self.job_settings.s3_location is not None:
                    copy_file_to_s3(
                        f,
                        f"{self.job_settings.s3_location}/{dst_file_path.name}"
                    )
                else:
                    dst_file_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(
                        f,
                        dst_dir / f.relative_to(self.job_settings.input_source)
                    )

            if self.job_settings.chronic_start_flag:
                # Copy the probe.json and binary_info.json files
                probe_json = self.job_settings.input_source / "probe.json"
                binary_info_json = (
                    self.job_settings.input_source / "binary_info.json"
                )
                if self.job_settings.s3_location is not None:
                    copy_file_to_s3(
                        probe_json,
                        f"{self.job_settings.s3_location}/{probe_json.name}"
                    )
                    copy_file_to_s3(
                        binary_info_json,
                        f"{self.job_settings.s3_location}/"
                        f"{binary_info_json.name}"
                    )
                else:
                    shutil.copy(probe_json, dst_dir / probe_json.name)
                    shutil.copy(
                        binary_info_json,
                        dst_dir / binary_info_json.name
                    )
        elif self.job_settings.reader_name == ReaderName.OPENEPHYS:
            patterns_to_ignore = ["*.dat"]
            if self.job_settings.s3_location is not None:
                # If we are uploading to S3, we don't copy the files
                # to the local directory, but just upload them directly
                sync_dir_to_s3(
                    self.job_settings.input_source,
                    self.job_settings.s3_location,
                    exclude=patterns_to_ignore
                )
            else:
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
                if self.job_settings.s3_location is not None:
                    dst_location = dst_raw_file.relative_to(
                        self.job_settings.output_directory
                    )
                    copy_file_to_s3(
                        dst_raw_file,
                        f"{self.job_settings.s3_location}/{dst_location}"
                    )
                    # remove local file after copying to S3
                    dst_raw_file.unlink()

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
            if self.job_settings.s3_location is not None:
                # If we are uploading to S3, we don't write the data to a
                # local directory, but just upload it directly
                zarr_path = (
                    f"{self.job_settings.s3_location}/{output_dir.name}/"
                    f"{experiment_name}_{stream_name}.zarr"
                )
            else:
                zarr_path = \
                    output_dir / f"{experiment_name}_{stream_name}.zarr"
                if (
                    platform.system() == "Windows"
                    and len(str(zarr_path)) > max_windows_filename_len
                ):
                    raise Exception(
                        f"File name for zarr path is too long "
                        f"({len(str(zarr_path))})"
                        f" and might lead to errors. Use a shorter "
                        f"destination path."
                    )
            # compression for times is disabled
            compressor_by_dataset = dict(times=None)

            if self.job_settings.reader_name == ReaderName.CHRONIC:
                from aind_ephys_transformation.compression_utils import (
                    write_or_append_recording_to_zarr,
                )

                # For Chronic data, we use a custom function to write the
                # recording to zarr, which handles the appending of data
                write_or_append_recording_to_zarr(
                    recording=rec,
                    folder_path=zarr_path,
                    compressor=compressor,
                    compressor_by_dataset=compressor_by_dataset,
                    annotations_to_update=["start_end_frames"],
                    **job_kwargs,
                )
            else:
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
        logging.info(
            "Copying and clipping source data. This may take a minute."
        )
        if self.job_settings.reader_name == ReaderName.CHRONIC:
            # For chronic data we are not clpping the data, so we save the
            # files directly to the output directory
            clipped_data_path = self.job_settings.output_directory
        else:
            # For OpenEphys data we clip the .dat files to a small number of
            # frames and save it to the "ecephys_clipped" directory
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
        # No need to scale the Chronic Onix recordings, only Open Ephys
        if self.job_settings.reader_name == ReaderName.OPENEPHYS:
            # Scale the OpenEphys recordings
            scaled_read_blocks = self._scale_read_blocks(
                read_blocks=read_blocks,
                random_seed=self.job_settings.random_seed,
                num_chunks_per_segment=(
                    self.job_settings.scale_num_chunks_per_segment
                ),
                chunk_size=self.job_settings.scale_chunk_size,
            )
        else:
            # For Chronic data, we don't scale the recordings
            scaled_read_blocks = read_blocks
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
