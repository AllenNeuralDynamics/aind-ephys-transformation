"""Tests for the ephys package"""

import json
import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from aind_data_transformation.core import JobResponse
from numcodecs import Blosc
from wavpack_numcodecs import WavPack

from aind_ephys_transformation.ephys_job import (
    EphysCompressionJob,
    EphysJobSettings,
)
from aind_ephys_transformation.models import CompressorName


TEST_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"
DATA_DIR = TEST_DIR / "v0.6.x_neuropixels_multiexp_multistream"


class TestEphysJob(unittest.TestCase):
    """Tests for ephys_job module"""

    @classmethod
    def setUpClass(cls):
        """Setup basic job settings and job that can be used across tests"""
        basic_job_settings = EphysJobSettings(
            input_source=DATA_DIR,
            output_directory=Path("output_dir"),
            compress_job_save_kwargs={"n_jobs": 1},
        )
        cls.basic_job_settings = basic_job_settings
        cls.basic_job = EphysCompressionJob(job_settings=basic_job_settings)

    @patch("warnings.warn")
    def test_get_compressor_default(self, _: MagicMock):
        """Tests _get_compressor_default method with default settings."""
        compressor = self.basic_job._get_compressor()
        expected_default = WavPack(
            bps=0,
            dynamic_noise_shaping=True,
            level=3,
            num_decoding_threads=8,
            num_encoding_threads=1,
            shaping_weight=0.0,
        )
        self.assertEqual(expected_default, compressor)

    @patch("warnings.warn")
    def test_get_compressor_wavpack(self, _: MagicMock):
        """Tests _get_compressor_default method with WavPack settings."""
        compressor_kwargs = {
            "level": 4,
        }
        settings = EphysJobSettings(
            input_source=Path("input_dir"),
            output_directory=Path("output_dir"),
            compressor_name=CompressorName.WAVPACK,
            compressor_kwargs=compressor_kwargs,
        )
        etl_job = EphysCompressionJob(job_settings=settings)
        compressor = etl_job._get_compressor()
        expected_compressor = WavPack(
            bps=0,
            dynamic_noise_shaping=True,
            level=4,
            num_decoding_threads=8,
            num_encoding_threads=1,
            shaping_weight=0.0,
        )
        self.assertEqual(expected_compressor, compressor)

    def test_get_compressor_blosc(self):
        """Tests _get_compressor_default method with Blosc settings."""
        compressor_kwargs = {
            "clevel": 4,
        }
        settings = EphysJobSettings(
            input_source=Path("input_dir"),
            output_directory=Path("output_dir"),
            compressor_name=CompressorName.BLOSC,
            compressor_kwargs=compressor_kwargs,
        )
        etl_job = EphysCompressionJob(job_settings=settings)
        compressor = etl_job._get_compressor()
        expected_compressor = Blosc(clevel=4)
        self.assertEqual(expected_compressor, compressor)

    def test_get_compressor_error(self):
        """Tests _get_compressor_default method with unknown compressor."""

        etl_job = EphysCompressionJob(
            job_settings=EphysJobSettings.model_construct(
                input_source=Path("input_dir"),
                output_directory=Path("output_dir"),
                compressor_name="UNKNOWN",
            )
        )
        with self.assertRaises(Exception) as e:
            etl_job._get_compressor()

        expected_error_message = (
            "Unknown compressor. Please select one of "
            "[<CompressorName.BLOSC: 'blosc'>, "
            "<CompressorName.WAVPACK: 'wavpack'>]",
        )
        self.assertEqual(expected_error_message, e.exception.args)

    def test_get_read_blocks(self):
        """Tests _get_read_blocks method"""
        read_blocks = self.basic_job._get_read_blocks()
        # Instead of constructing OpenEphysBinaryRecordingExtractor to
        # compare against, we can just compare the repr of the classes
        read_blocks_repr = []
        for read_block in read_blocks:
            copied_read_block = read_block
            copied_read_block["recording"] = repr(read_block["recording"])
            read_blocks_repr.append(copied_read_block)
        extractor_str_1 = (
            "OpenEphysBinaryRecordingExtractor: 8 channels - 30.0kHz "
            "- 1 segments - 100 samples \n"
            "                                   0.00s (3.33 ms) - int16 dtype "
            "- 1.56 KiB"
        )
        extractor_str_2 = (
            "OpenEphysBinaryRecordingExtractor: 384 channels - 30.0kHz "
            "- 1 segments - 100 samples \n"
            "                                   0.00s (3.33 ms) - int16 dtype "
            "- 75.00 KiB"
        )
        expected_read_blocks = [
            {
                "recording": extractor_str_1,
                "experiment_name": "experiment1",
                "stream_name": "Record Node 101#NI-DAQmx-103.PXIe-6341",
            },
            {
                "recording": extractor_str_2,
                "experiment_name": "experiment1",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeB",
            },
            {
                "recording": extractor_str_2,
                "experiment_name": "experiment1",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeC",
            },
            {
                "recording": extractor_str_1,
                "experiment_name": "experiment3",
                "stream_name": "Record Node 101#NI-DAQmx-103.PXIe-6341",
            },
            {
                "recording": extractor_str_2,
                "experiment_name": "experiment3",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeB",
            },
            {
                "recording": extractor_str_2,
                "experiment_name": "experiment3",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeC",
            },
            {
                "recording": extractor_str_1,
                "experiment_name": "experiment6",
                "stream_name": "Record Node 101#NI-DAQmx-103.PXIe-6341",
            },
            {
                "recording": extractor_str_2,
                "experiment_name": "experiment6",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeB",
            },
            {
                "recording": extractor_str_2,
                "experiment_name": "experiment6",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeC",
            },
        ]
        expected_scaled_read_blocks_str = set(
            [json.dumps(o) for o in expected_read_blocks]
        )
        read_blocks_repr_str = set([json.dumps(o) for o in read_blocks_repr])
        self.assertEqual(expected_scaled_read_blocks_str, read_blocks_repr_str)

    def test_scale_read_blocks(self):
        """Tests _scale_read_blocks method"""
        read_blocks = self.basic_job._get_read_blocks()
        scaled_read_blocks = self.basic_job._scale_read_blocks(
            read_blocks=read_blocks,
            random_seed=0,
            num_chunks_per_segment=10,
            chunk_size=50,
        )
        # Instead of constructing ScaledRecording classes to
        # compare against, we can just compare the repr of the classes
        scaled_read_blocks_repr = []
        for read_block in scaled_read_blocks:
            copied_read_block = read_block
            copied_read_block["scaled_recording"] = repr(
                read_block["scaled_recording"]
            )
            scaled_read_blocks_repr.append(copied_read_block)

        extractor_str_1 = (
            "OpenEphysBinaryRecordingExtractor: 8 channels - 30.0kHz "
            "- 1 segments - 100 samples \n"
            "                                   0.00s (3.33 ms) - int16 dtype "
            "- 1.56 KiB"
        )
        extractor_str_2 = (
            "ScaleRecording: 384 channels - 30.0kHz - 1 segments "
            "- 100 samples - 0.00s (3.33 ms) - int16 dtype \n"
            "                75.00 KiB"
        )

        expected_scaled_read_blocks = [
            {
                "scaled_recording": extractor_str_1,
                "experiment_name": "experiment1",
                "stream_name": "Record Node 101#NI-DAQmx-103.PXIe-6341",
            },
            {
                "scaled_recording": extractor_str_2,
                "experiment_name": "experiment1",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeB",
            },
            {
                "scaled_recording": extractor_str_2,
                "experiment_name": "experiment1",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeC",
            },
            {
                "scaled_recording": extractor_str_1,
                "experiment_name": "experiment3",
                "stream_name": "Record Node 101#NI-DAQmx-103.PXIe-6341",
            },
            {
                "scaled_recording": extractor_str_2,
                "experiment_name": "experiment3",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeB",
            },
            {
                "scaled_recording": extractor_str_2,
                "experiment_name": "experiment3",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeC",
            },
            {
                "scaled_recording": extractor_str_1,
                "experiment_name": "experiment6",
                "stream_name": "Record Node 101#NI-DAQmx-103.PXIe-6341",
            },
            {
                "scaled_recording": extractor_str_2,
                "experiment_name": "experiment6",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeB",
            },
            {
                "scaled_recording": extractor_str_2,
                "experiment_name": "experiment6",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeC",
            },
        ]
        expected_scaled_read_blocks_str = set(
            [json.dumps(o) for o in expected_scaled_read_blocks]
        )
        scaled_read_blocks_repr_str = set(
            [json.dumps(o) for o in scaled_read_blocks_repr]
        )
        self.assertEqual(
            expected_scaled_read_blocks_str, scaled_read_blocks_repr_str
        )

    def test_get_streams_to_clip(self):
        """Tests _get_streams_to_clip method"""
        streams_to_clip = self.basic_job._get_streams_to_clip()
        # TODO: If we want finer granularity, we can compare the numpy.memmap
        #  directly instead of just checking their shape
        streams_to_clip_just_shape = []
        for stream_to_clip in streams_to_clip:
            stream_to_clip_copy = {
                "relative_path_name": stream_to_clip["relative_path_name"],
                "n_chan": stream_to_clip["n_chan"],
                "data": stream_to_clip["data"].shape,
            }
            streams_to_clip_just_shape.append(stream_to_clip_copy)

        def base_path(num: int) -> Path:
            """Utility method to construct expected output base paths"""
            return (
                Path("Record Node 101")
                / f"experiment{num}"
                / "recording1"
                / "continuous"
            )

        expected_output = [
            {
                "relative_path_name": str(
                    base_path(1) / "Neuropix-PXI-100.ProbeB" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(1) / "Neuropix-PXI-100.ProbeC" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(1) / "NI-DAQmx-103.PXIe-6341" / "continuous.dat"
                ),
                "n_chan": 8,
                "data": (100, 8),
            },
            {
                "relative_path_name": str(
                    base_path(6) / "Neuropix-PXI-100.ProbeB" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(6) / "Neuropix-PXI-100.ProbeC" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(6) / "NI-DAQmx-103.PXIe-6341" / "continuous.dat"
                ),
                "n_chan": 8,
                "data": (100, 8),
            },
            {
                "relative_path_name": str(
                    base_path(3) / "NI-DAQmx-103.PXIe-6341" / "continuous.dat"
                ),
                "n_chan": 8,
                "data": (100, 8),
            },
            {
                "relative_path_name": str(
                    base_path(3) / "Neuropix-PXI-100.ProbeB" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(3) / "Neuropix-PXI-100.ProbeC" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
        ]
        expected_output_str = set([json.dumps(o) for o in expected_output])
        streams_to_clip_just_shape_str = set(
            [json.dumps(o) for o in streams_to_clip_just_shape]
        )
        self.assertEqual(expected_output_str, streams_to_clip_just_shape_str)

    @patch("shutil.copytree")
    @patch("shutil.ignore_patterns")
    @patch("numpy.memmap")
    def test_copy_and_clip_data(
        self,
        mock_memmap: MagicMock,
        mock_ignore_patterns: MagicMock,
        mock_copy_tree: MagicMock,
    ):
        """Tests _copy_and_clip_data method"""
        mock_ignore_patterns.return_value = ["*.dat"]

        def base_path(num: int) -> Path:
            """Utility method to construct expected output base paths"""
            return (
                Path("Record Node 101")
                / f"experiment{num}"
                / "recording1"
                / "continuous"
            )

        expected_output = [
            {
                "relative_path_name": str(
                    base_path(1) / "Neuropix-PXI-100.ProbeB" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(1) / "Neuropix-PXI-100.ProbeC" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(1) / "NI-DAQmx-103.PXIe-6341" / "continuous.dat"
                ),
                "n_chan": 8,
                "data": (100, 8),
            },
            {
                "relative_path_name": str(
                    base_path(6) / "Neuropix-PXI-100.ProbeB" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(6) / "Neuropix-PXI-100.ProbeC" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(6) / "NI-DAQmx-103.PXIe-6341" / "continuous.dat"
                ),
                "n_chan": 8,
                "data": (100, 8),
            },
            {
                "relative_path_name": str(
                    base_path(3) / "NI-DAQmx-103.PXIe-6341" / "continuous.dat"
                ),
                "n_chan": 8,
                "data": (100, 8),
            },
            {
                "relative_path_name": str(
                    base_path(3) / "Neuropix-PXI-100.ProbeB" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
            {
                "relative_path_name": str(
                    base_path(3) / "Neuropix-PXI-100.ProbeC" / "continuous.dat"
                ),
                "n_chan": 384,
                "data": (100, 384),
            },
        ]
        expected_memmap_calls = []
        for foobar in expected_output:
            expected_memmap_calls.append(
                call(
                    filename=Path(foobar["relative_path_name"]),
                    dtype="int16",
                    shape=foobar["data"],
                    order="C",
                    mode="w+",
                )
            )
            expected_memmap_calls.append(
                call().__setitem__(slice(None, None, None), foobar["data"])
            )
        self.basic_job._copy_and_clip_data(
            dst_dir=Path("."), stream_gen=expected_output
        )
        mock_ignore_patterns.assert_called_once_with("*.dat")
        mock_copy_tree.assert_called_once_with(
            DATA_DIR, Path("."), ignore=["*.dat"]
        )
        mock_memmap.assert_has_calls(expected_memmap_calls)

    @patch("warnings.warn")
    @patch(
        "spikeinterface.extractors.neoextractors.openephys"
        ".OpenEphysBinaryRecordingExtractor.save"
    )
    @patch("spikeinterface.preprocessing.normalize_scale.ScaleRecording.save")
    def test_compress_and_write_scaled_blocks(
        self,
        mock_scale_save: MagicMock,
        mock_bin_save: MagicMock,
        _: MagicMock,
    ):
        """Tests _compress_and_write_block method with scaled rec"""
        read_blocks = self.basic_job._get_read_blocks()
        scaled_read_blocks = self.basic_job._scale_read_blocks(
            read_blocks=read_blocks,
            random_seed=0,
            num_chunks_per_segment=10,
            chunk_size=50,
        )
        compressor = self.basic_job._get_compressor()
        max_windows_filename_len = (
            self.basic_job.job_settings.compress_max_windows_filename_len
        )
        output_dir = (
            self.basic_job.job_settings.output_directory / "compressed"
        )
        output_format = (
            self.basic_job.job_settings.compress_write_output_format
        )
        job_kwargs = self.basic_job.job_settings.compress_job_save_kwargs
        self.basic_job._compress_and_write_block(
            read_blocks=scaled_read_blocks,
            compressor=compressor,
            max_windows_filename_len=max_windows_filename_len,
            output_dir=output_dir,
            output_format=output_format,
            job_kwargs=job_kwargs,
        )
        mock_bin_save.assert_has_calls(
            [
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment1_Record Node 101#NI-DAQmx-103"
                            ".PXIe-6341.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment3_Record Node 101#NI-DAQmx-103"
                            ".PXIe-6341.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment6_Record Node 101#NI-DAQmx-103"
                            ".PXIe-6341.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
            ]
        )
        mock_scale_save.assert_has_calls(
            [
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment1_Record Node 101#Neuropix-PXI-100"
                            ".ProbeB.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment1_Record Node 101#Neuropix-PXI-100"
                            ".ProbeC.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment3_Record Node 101#Neuropix-PXI-100"
                            ".ProbeB.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment3_Record Node 101#Neuropix-PXI-100"
                            ".ProbeC.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment6_Record Node 101#Neuropix-PXI-100"
                            ".ProbeB.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment6_Record Node 101#Neuropix-PXI-100"
                            ".ProbeC.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
            ]
        )

    @patch("warnings.warn")
    @patch(
        "spikeinterface.extractors.neoextractors.openephys"
        ".OpenEphysBinaryRecordingExtractor.save"
    )
    def test_compress_and_write_read_blocks(
        self,
        mock_bin_save: MagicMock,
        _: MagicMock,
    ):
        """Tests _compress_and_write_block method with raw rec"""
        read_blocks = self.basic_job._get_read_blocks()
        compressor = self.basic_job._get_compressor()
        max_windows_filename_len = (
            self.basic_job.job_settings.compress_max_windows_filename_len
        )
        output_dir = (
            self.basic_job.job_settings.output_directory / "compressed"
        )
        output_format = (
            self.basic_job.job_settings.compress_write_output_format
        )
        job_kwargs = self.basic_job.job_settings.compress_job_save_kwargs
        self.basic_job._compress_and_write_block(
            read_blocks=read_blocks,
            compressor=compressor,
            max_windows_filename_len=max_windows_filename_len,
            output_dir=output_dir,
            output_format=output_format,
            job_kwargs=job_kwargs,
        )
        mock_bin_save.assert_has_calls(
            [
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment1_Record Node 101#NI-DAQmx-103"
                            ".PXIe-6341.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment1_Record Node 101#Neuropix-PXI-100"
                            ".ProbeB.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment1_Record Node 101#Neuropix-PXI-100"
                            ".ProbeC.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment3_Record Node 101#NI-DAQmx-103"
                            ".PXIe-6341.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment3_Record Node 101#Neuropix-PXI-100"
                            ".ProbeB.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment3_Record Node 101#Neuropix-PXI-100"
                            ".ProbeC.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment6_Record Node 101#NI-DAQmx-103"
                            ".PXIe-6341.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment6_Record Node 101#Neuropix-PXI-100"
                            ".ProbeB.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
                call(
                    format="zarr",
                    folder=(
                        Path("output_dir")
                        / "compressed"
                        / (
                            "experiment6_Record Node 101#Neuropix-PXI-100"
                            ".ProbeC.zarr"
                        )
                    ),
                    compressor=WavPack(
                        bps=0,
                        dynamic_noise_shaping=True,
                        level=3,
                        num_decoding_threads=8,
                        num_encoding_threads=1,
                        shaping_weight=0.0,
                    ),
                    compressor_by_dataset={"times": None},
                    n_jobs=1,
                ),
            ],
            any_order=True,
        )

    @patch("os.cpu_count")
    @patch("warnings.warn")
    @patch(
        "spikeinterface.extractors.neoextractors.openephys"
        ".OpenEphysBinaryRecordingExtractor.save"
    )
    def test_compress_and_write_read_blocks_cpu_count(
        self,
        mock_bin_save: MagicMock,
        _: MagicMock,
        mock_os_cpu_count: MagicMock,
    ):
        """Tests _compress_and_write_block method with n_jobs set to -1"""
        mock_os_cpu_count.return_value = 1
        read_blocks = self.basic_job._get_read_blocks()
        compressor = self.basic_job._get_compressor()
        max_windows_filename_len = (
            self.basic_job.job_settings.compress_max_windows_filename_len
        )
        output_dir = (
            self.basic_job.job_settings.output_directory / "compressed"
        )
        output_format = (
            self.basic_job.job_settings.compress_write_output_format
        )
        self.basic_job._compress_and_write_block(
            read_blocks=read_blocks,
            compressor=compressor,
            max_windows_filename_len=max_windows_filename_len,
            output_dir=output_dir,
            output_format=output_format,
            job_kwargs={"n_jobs": -1},
        )
        self.assertEqual(9, len(mock_bin_save.mock_calls))

    @patch("warnings.warn")
    @patch(
        "spikeinterface.extractors.neoextractors.openephys"
        ".OpenEphysBinaryRecordingExtractor.save"
    )
    @patch("platform.system")
    def test_compress_and_write_windows_filename_error(
        self,
        mock_platform: MagicMock,
        mock_bin_save: MagicMock,
        _: MagicMock,
    ):
        """Tests _compress_and_write_block method when filename is too long
        for Windows OS"""
        mock_platform.return_value = "Windows"
        read_blocks = self.basic_job._get_read_blocks()
        compressor = self.basic_job._get_compressor()
        max_windows_filename_len = 100
        output_dir = Path("x" * 100)
        output_format = (
            self.basic_job.job_settings.compress_write_output_format
        )
        job_kwargs = self.basic_job.job_settings.compress_job_save_kwargs
        with self.assertRaises(Exception) as e:
            self.basic_job._compress_and_write_block(
                read_blocks=read_blocks,
                compressor=compressor,
                max_windows_filename_len=max_windows_filename_len,
                output_dir=output_dir,
                output_format=output_format,
                job_kwargs=job_kwargs,
            )
        self.assertEqual(
            (
                "File name for zarr path is too long (156) and might lead "
                "to errors. Use a shorter destination path.",
            ),
            e.exception.args,
        )
        mock_bin_save.assert_not_called()

    @patch("warnings.warn")
    @patch(
        "aind_ephys_transformation.ephys_job.EphysCompressionJob"
        "._compress_and_write_block"
    )
    @patch(
        "aind_ephys_transformation.ephys_job.EphysCompressionJob"
        "._copy_and_clip_data"
    )
    @patch("logging.info")
    def test_compress_raw_data(
        self,
        mock_log_info: MagicMock,
        mock_copy_and_clip_data: MagicMock,
        mock_compress_and_write_block: MagicMock,
        _: MagicMock,
    ):
        """Tests _compress_raw_data method with basic job"""
        self.basic_job._compress_raw_data()
        settings1_path = DATA_DIR / "Record Node 101" / "settings.xml"
        settings3_path = DATA_DIR / "Record Node 101" / "settings_3.xml"
        settings6_path = DATA_DIR / "Record Node 101" / "settings_6.xml"
        mock_log_info.assert_has_calls(
            [
                call("Clipping source data. This may take a minute."),
                call("Finished clipping source data."),
                call("Compressing source data."),
                call("Finished compressing source data."),
            ],
            any_order=True,
        )
        self.assertEqual(1, len(mock_copy_and_clip_data.mock_calls))
        # More granularity can be added in the future. For now, we just compare
        # the length of the stream_gen list
        actual_clip_args_derived = mock_copy_and_clip_data.mock_calls[0].kwargs
        actual_clip_args_derived["stream_gen"] = len(
            list(actual_clip_args_derived["stream_gen"])
        )
        expected_clip_args_derived = {
            "stream_gen": 9,
            "dst_dir": Path("output_dir") / "ecephys_clipped",
        }
        self.assertEqual(expected_clip_args_derived, actual_clip_args_derived)

        self.assertEqual(1, len(mock_compress_and_write_block.mock_calls))
        # More granularity can be added in the future. For now, we just compare
        # the length of the read_blocks list
        actual_comp_args_derived = mock_compress_and_write_block.mock_calls[
            0
        ].kwargs
        actual_comp_args_derived["read_blocks"] = len(
            list(actual_comp_args_derived["read_blocks"])
        )
        expected_comp_args_derived = {
            "read_blocks": 9,
            "compressor": WavPack(
                bps=0,
                dynamic_noise_shaping=True,
                level=3,
                num_decoding_threads=8,
                num_encoding_threads=1,
                shaping_weight=0.0,
            ),
            "max_windows_filename_len": 150,
            "output_dir": Path("output_dir") / "ecephys_compressed",
            "output_format": "zarr",
            "job_kwargs": {"n_jobs": 1},
        }
        self.assertEqual(expected_comp_args_derived, actual_comp_args_derived)

    @patch("aind_ephys_transformation.ephys_job.datetime")
    @patch(
        "aind_ephys_transformation.ephys_job.EphysCompressionJob"
        "._compress_raw_data"
    )
    def test_run_job(
        self, mock_compress_raw_data: MagicMock, mock_datetime: MagicMock
    ):
        """Tests run_job method"""
        mock_start_time = datetime(2020, 10, 10, 1, 30, 0)
        mock_end_time = datetime(2020, 10, 10, 5, 25, 17)
        mock_time_delta = mock_end_time - mock_start_time
        mock_datetime.now.side_effect = [
            datetime(2020, 10, 10, 1, 30, 0),
            datetime(2020, 10, 10, 5, 25, 17),
        ]
        job_response = self.basic_job.run_job()
        expected_job_response = JobResponse(
            status_code=200,
            message=f"Job finished in: {mock_time_delta}",
            data=None,
        )
        self.assertEqual(expected_job_response, job_response)
        mock_compress_raw_data.assert_called_once()


if __name__ == "__main__":
    unittest.main()
