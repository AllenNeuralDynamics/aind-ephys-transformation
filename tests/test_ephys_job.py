"""Tests for the ephys package"""

import json
import os
import unittest
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from numcodecs import Blosc
from wavpack_numcodecs import WavPack

from spikeinterface import load
from spikeinterface.core.testing import check_recordings_equal

from aind_data_transformation.core import JobResponse
from aind_ephys_transformation.ephys_job import (
    EphysCompressionJob,
    EphysJobSettings,
)
from aind_ephys_transformation.models import CompressorName

TEST_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"
OE_DATA_DIR = TEST_DIR / "v0.6.x_neuropixels_multiexp_multistream"
OE_DATA_DIR_NOT_ALIGNED = TEST_DIR / "v0.6.x_neuropixels_not_aligned"
CHRONIC_DATA_DIR = TEST_DIR / "chronic-test-data"


class TestEphysJob(unittest.TestCase):
    """Tests for ephys_job module"""

    @classmethod
    def setUpClass(cls):
        """Setup basic job settings and job that can be used across tests"""
        basic_job_settings = EphysJobSettings(
            input_source=OE_DATA_DIR,
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
            expected_scaled_read_blocks_str,
            scaled_read_blocks_repr_str,
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
            OE_DATA_DIR, Path("."), ignore=["*.dat"]
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
        mock_log_info.assert_has_calls(
            [
                call("Checking timestamps alignment."),
                call(
                    "Copying and clipping source data. This may take a minute."
                ),
                call("Finished copying and clipping source data."),
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
        self,
        mock_compress_raw_data: MagicMock,
        mock_datetime: MagicMock,
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

    @patch("pathlib.Path.unlink")
    @patch("numpy.memmap")
    @patch("aind_ephys_transformation.ephys_job.sync_dir_to_s3")
    @patch("aind_ephys_transformation.ephys_job.copy_file_to_s3")
    def test_s3_location_copy_and_clip(
        self,
        mock_copy_file_to_s3: MagicMock,
        mock_sync_dir_to_s3: MagicMock,
        mock_memmap: MagicMock,
        _: MagicMock,
    ):
        """Tests S3 location generation for OpenEphys data"""
        job_settings_s3 = EphysJobSettings(
            input_source=OE_DATA_DIR,
            output_directory=Path("output_dir_s3"),
            compress_job_save_kwargs={"n_jobs": 1},
            s3_location="s3://bucket/session/",
        )
        # s3_location will be "s3://bucket/session" after validation
        expected_s3_base = "s3://bucket/session"

        job_s3 = EphysCompressionJob(job_settings=job_settings_s3)

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
        expected_copy_calls = []
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
            expected_copy_calls.append(
                call(
                    Path(foobar["relative_path_name"]),
                    f"{expected_s3_base}/{foobar['relative_path_name']}",
                )
            )
        dst_dir = Path(".")
        job_s3._copy_and_clip_data(dst_dir=dst_dir, stream_gen=expected_output)

        # Verify sync_dir_to_s3 was called correctly for non-.dat files
        mock_sync_dir_to_s3.assert_called_once_with(
            OE_DATA_DIR, expected_s3_base, exclude=["*.dat"]
        )

        # Verify copy_file_to_s3 was called for each .dat file
        expected_copy_calls = []
        for foobar in expected_output:
            expected_copy_calls.append(
                call(
                    dst_dir / foobar["relative_path_name"],
                    f"{expected_s3_base}/{foobar['relative_path_name']}",
                )
            )
        mock_copy_file_to_s3.assert_has_calls(
            expected_copy_calls, any_order=True
        )


class TestCheckTimeAlignment(unittest.TestCase):
    """Tests time alignment check"""

    @classmethod
    def setUpClass(cls):
        """Setup basic job settings and job that can be used across tests."""
        basic_job_settings_raise = EphysJobSettings(
            input_source=OE_DATA_DIR_NOT_ALIGNED,
            output_directory=Path("output_dir_align"),
            compress_job_save_kwargs={"n_jobs": 1},
        )
        cls.basic_job_settings_raise = basic_job_settings_raise
        cls.basic_job_raise = EphysCompressionJob(
            job_settings=basic_job_settings_raise
        )

        basic_job_settings_warn = EphysJobSettings(
            input_source=OE_DATA_DIR_NOT_ALIGNED,
            output_directory=Path("output_dir_align"),
            compress_job_save_kwargs={"n_jobs": 1},
            check_timestamps=False,
        )
        cls.basic_job_settings_warn = basic_job_settings_warn
        cls.basic_job_warn = EphysCompressionJob(
            job_settings=basic_job_settings_warn
        )

    def test_check_alignment(self):
        """Tests check_time_alignment returns False"""
        timestamps_ok = self.basic_job_raise._check_timestamps_alignment()
        self.assertFalse(timestamps_ok)

    def test_job_failure(self):
        """Tests _compress_raw_data method"""
        with self.assertRaises(Exception) as e:
            self.basic_job_raise._compress_raw_data()
        self.assertEqual(
            (
                "Timestamps are not aligned. Please align timestamps "
                "using aind-ephys-rig-qc before compressing the data.",
            ),
            e.exception.args,
        )

    @patch("logging.warning")
    @patch(
        "aind_ephys_transformation.ephys_job.EphysCompressionJob"
        "._compress_and_write_block"
    )
    @patch(
        "aind_ephys_transformation.ephys_job.EphysCompressionJob"
        "._copy_and_clip_data"
    )
    def test_job_warning(
        self,
        mock_copy_and_clip_data: MagicMock,
        mock_compress_raw_data: MagicMock,
        mock_logger: MagicMock,
    ):
        """Tests _compress_raw_data method"""
        self.basic_job_warn._compress_raw_data()
        mock_logger.assert_called_with(
            "Timestamps are not aligned, but timestamps check is "
            "disabled. Proceeding with compression.",
        )


class TestChronicCompressJob(unittest.TestCase):
    """Tests for the EphysCompressionJob with a chronic compress job"""

    @classmethod
    def setUpClass(cls):
        """Setup basic job settings and job that can be used across tests"""
        chronic_job_settings = EphysJobSettings(
            input_source=CHRONIC_DATA_DIR,
            output_directory=Path("output_dir_chronic"),
            compress_job_save_kwargs={"n_jobs": 1},
            reader_name="chronic",
            chronic_start_flag=True,
        )
        cls.chronic_job_settings = chronic_job_settings
        cls.chronic_job = EphysCompressionJob(
            job_settings=chronic_job_settings
        )

        chronic_job_settings_append1 = EphysJobSettings(
            input_source=CHRONIC_DATA_DIR,
            output_directory=Path("output_dir_chronic_append"),
            compress_job_save_kwargs={"n_jobs": 1},
            chronic_chunks_to_compress=["2025-05-13T19-00-00"],
            reader_name="chronic",
            chronic_start_flag=True,
        )
        cls.chronic_job_settings_append1 = chronic_job_settings_append1
        cls.chronic_job_append1 = EphysCompressionJob(
            job_settings=chronic_job_settings_append1
        )

        chronic_job_settings_append2 = EphysJobSettings(
            input_source=CHRONIC_DATA_DIR,
            output_directory=Path("output_dir_chronic_append"),
            compress_job_save_kwargs={"n_jobs": 1},
            chronic_chunks_to_compress=[
                "2025-05-13T20-00-00",
                "2025-05-13T21-00-00",
            ],
            reader_name="chronic",
        )
        cls.chronic_job_settings_append2 = chronic_job_settings_append2
        cls.chronic_job_append2 = EphysCompressionJob(
            job_settings=chronic_job_settings_append2
        )

        chronic_job_settings_ns = EphysJobSettings(
            input_source=CHRONIC_DATA_DIR,
            output_directory=Path("output_dir_chronic_ns"),
            compress_job_save_kwargs={"n_jobs": 1},
            chronic_chunks_to_compress=[
                "2025-05-13T19-00-00",
                "2025-05-13T21-00-00",
            ],
            reader_name="chronic",
        )
        cls.chronic_job_settings_ns = chronic_job_settings_ns
        cls.chronic_job_ns = EphysCompressionJob(
            job_settings=chronic_job_settings_ns
        )

        chronic_job_settings_filter = EphysJobSettings(
            input_source=CHRONIC_DATA_DIR,
            output_directory=Path("output_dir_chronic_filter"),
            compress_job_save_kwargs={"n_jobs": 1},
            reader_name="chronic",
            chronic_chunks_to_compress=["2025-05-13T19-00-00"],
        )
        cls.chronic_job_settings_filter = chronic_job_settings_filter
        cls.chronic_job_filter = EphysCompressionJob(
            job_settings=chronic_job_settings_filter
        )

        chronic_job_settings_no_match = EphysJobSettings(
            input_source=CHRONIC_DATA_DIR,
            output_directory=Path("output_dir_chronic_no_match"),
            compress_job_save_kwargs={"n_jobs": 1},
            reader_name="chronic",
            chronic_chunks_to_compress=["2024-05-13T19-00-00"],
        )
        cls.chronic_job_settings_no_match = chronic_job_settings_no_match
        cls.chronic_job_no_match = EphysCompressionJob(
            job_settings=chronic_job_settings_no_match
        )

        chronic_job_settings_multi_match = EphysJobSettings(
            input_source=CHRONIC_DATA_DIR,
            output_directory=Path("output_dir_chronic_multi_match"),
            compress_job_save_kwargs={"n_jobs": 1},
            reader_name="chronic",
            chronic_chunks_to_compress=["2025-05-13"],
        )
        cls.chronic_job_settings_multi_match = chronic_job_settings_multi_match
        cls.chronic_job_multi_match = EphysCompressionJob(
            job_settings=chronic_job_settings_multi_match
        )

    @classmethod
    def tearDownClass(cls):
        """Remove output directories created during tests"""
        for job in [
            cls.chronic_job,
            cls.chronic_job_append1,
            cls.chronic_job_append2,
            cls.chronic_job_ns,
            cls.chronic_job_filter,
            cls.chronic_job_no_match,
            cls.chronic_job_multi_match,
        ]:
            output_dir = job.job_settings.output_directory
            if Path(output_dir).exists():
                shutil.rmtree(output_dir)

    def test_get_read_blocks(self):
        """Tests _get_read_blocks method"""
        read_blocks = self.chronic_job._get_read_blocks()
        # Instead of constructing OpenEphysBinaryRecordingExtractor to
        # compare against, we can just compare the repr of the classes
        read_blocks_repr = []
        for read_block in read_blocks:
            copied_read_block = read_block
            print(repr(read_block["recording"]))
            copied_read_block["recording"] = repr(read_block["recording"])
            read_blocks_repr.append(copied_read_block)
        extractor_str = (
            "ConcatenateSegmentRecording: 384 channels - 30.0kHz - 1 segments "
            "- 300 samples - 0.01s (10.00 ms) \n                             "
            "int16 dtype - 225.00 KiB"
        )
        expected_read_blocks = [
            {
                "recording": extractor_str,
                "experiment_name": "experiment1",
                "stream_name": "AmplifierData",
            }
        ]
        expected_scaled_read_blocks_str = set(
            [json.dumps(o) for o in expected_read_blocks]
        )
        read_blocks_repr_str = set([json.dumps(o) for o in read_blocks_repr])
        self.assertEqual(expected_scaled_read_blocks_str, read_blocks_repr_str)

    def test_get_read_blocks_filter(self):
        """Tests _get_read_blocks method"""
        read_blocks = self.chronic_job_filter._get_read_blocks()
        # Instead of constructing OpenEphysBinaryRecordingExtractor to
        # compare against, we can just compare the repr of the classes
        read_blocks_repr = []
        for read_block in read_blocks:
            copied_read_block = read_block
            copied_read_block["recording"] = repr(read_block["recording"])
            read_blocks_repr.append(copied_read_block)
        extractor_str = (
            "ConcatenateSegmentRecording: 384 channels - 30.0kHz - 1 segments "
            "- 100 samples - 0.00s (3.33 ms) \n                             "
            "int16 dtype - 75.00 KiB"
        )
        expected_read_blocks = [
            {
                "recording": extractor_str,
                "experiment_name": "experiment1",
                "stream_name": "AmplifierData",
            }
        ]
        expected_scaled_read_blocks_str = set(
            [json.dumps(o) for o in expected_read_blocks]
        )
        read_blocks_repr_str = set([json.dumps(o) for o in read_blocks_repr])
        self.assertEqual(expected_scaled_read_blocks_str, read_blocks_repr_str)

    def test_read_blocks_no_match(self):
        """Tests _get_read_blocks method with no matching chunks"""
        with self.assertRaises(ValueError):
            list(self.chronic_job_no_match._get_read_blocks())

    def test_read_blocks_multi_match(self):
        """Tests _get_read_blocks method with multiple matching chunks"""
        with self.assertRaises(ValueError):
            list(self.chronic_job_multi_match._get_read_blocks())

    def test_get_streams_to_clip(self):
        """Tests _get_streams_to_clip method"""
        streams_to_clip = self.chronic_job._get_streams_to_clip()
        # TODO: If we want finer granularity, we can compare the numpy.memmap
        #  directly instead of just checking their shape
        assert len(list(streams_to_clip)) == 0

    @patch("shutil.copy")
    def test_copy_and_clip_data(
        self,
        mock_copy: MagicMock,
    ):
        """Tests _copy_and_clip_data method"""
        self.chronic_job._copy_and_clip_data(
            dst_dir=Path("."), stream_gen=iter([])
        )
        all_files_to_copy = [
            p
            for p in self.chronic_job.job_settings.input_source.glob("**/*")
            if p.is_file() and "AmplifierData" not in p.name
        ]
        expected_copy_calls = [
            call(
                p,
                Path(".")
                / p.relative_to(self.chronic_job.job_settings.input_source),
            )
            for p in all_files_to_copy
        ]
        mock_copy.assert_has_calls(expected_copy_calls, any_order=True)

    @patch("warnings.warn")
    def test_compress_and_write_read_blocks(
        self,
        _: MagicMock,
    ):
        """Tests _compress_and_write_block method with raw rec"""
        read_blocks = self.chronic_job._get_read_blocks()
        compressor = self.chronic_job._get_compressor()
        output_dir = (
            self.chronic_job.job_settings.output_directory / "compressed"
        )
        output_format = (
            self.chronic_job.job_settings.compress_write_output_format
        )
        max_windows_filename_len = (
            self.chronic_job.job_settings.compress_max_windows_filename_len
        )
        job_kwargs = self.chronic_job.job_settings.compress_job_save_kwargs
        self.chronic_job._compress_and_write_block(
            read_blocks=read_blocks,
            compressor=compressor,
            output_dir=output_dir,
            max_windows_filename_len=max_windows_filename_len,
            output_format=output_format,
            job_kwargs=job_kwargs,
        )

        # read output recording against recording to write
        read_blocks = self.chronic_job._get_read_blocks()
        zarr_files = [p for p in output_dir.glob("**/*.zarr")]
        self.assertEqual(1, len(zarr_files))

        recording_to_write = list(read_blocks)[0]["recording"]
        recording_loaded = load(zarr_files[0])
        check_recordings_equal(recording_loaded, recording_to_write)

    @patch("warnings.warn")
    def test_compress_and_append_read_blocks(
        self,
        _: MagicMock,
    ):
        """Tests _compress_and_write_block method with raw rec"""
        read_blocks1 = self.chronic_job_append1._get_read_blocks()
        compressor = self.chronic_job_append1._get_compressor()
        output_dir = (
            self.chronic_job_append1.job_settings.output_directory
            / "compressed"
        )
        output_format = (
            self.chronic_job.job_settings.compress_write_output_format
        )
        max_windows_filename_len = (
            self.chronic_job.job_settings.compress_max_windows_filename_len
        )
        job_kwargs = self.chronic_job.job_settings.compress_job_save_kwargs
        self.chronic_job_append1._compress_and_write_block(
            read_blocks=read_blocks1,
            compressor=compressor,
            output_dir=output_dir,
            max_windows_filename_len=max_windows_filename_len,
            output_format=output_format,
            job_kwargs=job_kwargs,
        )

        read_blocks2 = self.chronic_job_append2._get_read_blocks()
        self.chronic_job_append2._compress_and_write_block(
            read_blocks=read_blocks2,
            compressor=compressor,
            output_dir=output_dir,
            max_windows_filename_len=max_windows_filename_len,
            output_format=output_format,
            job_kwargs=job_kwargs,
        )

        # read output recording against recording to write
        read_blocks_single = self.chronic_job._get_read_blocks()
        zarr_files = [p for p in output_dir.glob("**/*.zarr")]
        self.assertEqual(1, len(zarr_files))

        recording_to_write = list(read_blocks_single)[0]["recording"]
        recording_loaded = load(zarr_files[0])
        check_recordings_equal(recording_loaded, recording_to_write)

    def test_non_consecutive_chunks(self):
        """Tests _skip_chunk method"""
        with self.assertRaises(ValueError):
            list(self.chronic_job_ns._get_read_blocks())

        self.chronic_job_ns.job_settings.check_chronic_consecutive_hours = (
            False
        )
        with self.assertWarns(UserWarning):
            # This should not raise an error,
            # but warn about non-consecutive chunks
            list(self.chronic_job_ns._get_read_blocks())

    def test_time_alignment_chronic(self):
        """Tests time alignment check for chronic data"""
        timestamps_ok = self.chronic_job._check_timestamps_alignment()
        self.assertTrue(
            timestamps_ok,
            "Timestamps should be aligned for chronic data.",
        )

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
        self.chronic_job._compress_raw_data()
        mock_log_info.assert_has_calls(
            [
                call("Checking timestamps alignment."),
                call(
                    "Copying and clipping source data. This may take a minute."
                ),
                call("Finished copying and clipping source data."),
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
            "stream_gen": 0,
            "dst_dir": self.chronic_job.job_settings.output_directory,
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
            "read_blocks": 1,
            "compressor": WavPack(
                bps=0,
                dynamic_noise_shaping=True,
                level=3,
                num_decoding_threads=8,
                num_encoding_threads=1,
                shaping_weight=0.0,
            ),
            "max_windows_filename_len": 150,
            "output_dir": self.chronic_job.job_settings.output_directory
            / "ecephys_compressed",
            "output_format": "zarr",
            "job_kwargs": {"n_jobs": 1},
        }
        self.assertEqual(expected_comp_args_derived, actual_comp_args_derived)

    @patch(
        "aind_ephys_transformation.compression_utils."
        "write_or_append_recording_to_zarr"
    )
    @patch("aind_ephys_transformation.ephys_job.sync_dir_to_s3")
    @patch("aind_ephys_transformation.ephys_job.copy_file_to_s3")
    def test_s3_location(
        self,
        mock_copy_file_to_s3: MagicMock,
        mock_sync_dir_to_s3: MagicMock,
        mock_write_or_append_recording_to_zarr: MagicMock,
    ):
        """Tests S3 location generation for chronic recordings"""
        job_settings_s3 = EphysJobSettings(
            input_source=CHRONIC_DATA_DIR,
            output_directory=Path("output_dir_s3"),
            compress_job_save_kwargs={"n_jobs": 1},
            s3_location="s3://bucket/session/",
            reader_name="chronic",
            chronic_start_flag=True,
        )
        # s3_location will be "s3://bucket/session" after validation
        expected_s3_base = "s3://bucket/session"

        chronic_job_s3 = EphysCompressionJob(job_settings=job_settings_s3)

        chronic_job_s3._compress_raw_data()

        mock_sync_dir_to_s3.assert_not_called()

        # Assert calls to copy_file_to_s3
        onix_ephys_dir = CHRONIC_DATA_DIR / "OnixEphys"
        expected_copy_calls = [
            call(
                onix_ephys_dir / "OnixEphys_Clock_2025-05-13T19-00-00.bin",
                f"{expected_s3_base}/OnixEphys/"
                "OnixEphys_Clock_2025-05-13T19-00-00.bin",
            ),
            call(
                onix_ephys_dir / "OnixEphys_Clock_2025-05-13T20-00-00.bin",
                f"{expected_s3_base}/OnixEphys/"
                "OnixEphys_Clock_2025-05-13T20-00-00.bin",
            ),
            call(
                onix_ephys_dir / "OnixEphys_Clock_2025-05-13T21-00-00.bin",
                f"{expected_s3_base}/OnixEphys/"
                "OnixEphys_Clock_2025-05-13T21-00-00.bin",
            ),
            call(
                CHRONIC_DATA_DIR / "probe.json",
                f"{expected_s3_base}/probe.json",
            ),
            call(
                CHRONIC_DATA_DIR / "binary_info.json",
                f"{expected_s3_base}/binary_info.json",
            ),
        ]
        mock_copy_file_to_s3.assert_has_calls(
            expected_copy_calls, any_order=True
        )
        self.assertEqual(mock_copy_file_to_s3.call_count, 5)

        # Assert call to write_or_append_recording_to_zarr
        expected_zarr_s3_path = (
            f"{expected_s3_base}/ecephys_compressed/"
            f"experiment1_AmplifierData.zarr"
        )

        mock_write_or_append_recording_to_zarr.assert_called_once()

        # Get the arguments of the call to the mock
        # call_args is a tuple (pos_args, named_args)
        _args, kwargs = mock_write_or_append_recording_to_zarr.call_args

        self.assertEqual(kwargs.get("folder_path"), expected_zarr_s3_path)
        self.assertIn("recording", kwargs)
        self.assertIsInstance(kwargs.get("compressor"), WavPack)
        self.assertEqual(kwargs.get("compressor_by_dataset"), {"times": None})
        self.assertEqual(
            kwargs.get("annotations_to_update"), ["start_end_frames"]
        )
        # n_jobs comes from job_settings.compress_job_save_kwargs
        self.assertEqual(kwargs.get("n_jobs"), 1)

    def test_wrong_s3(self):
        """Tests S3 location generation with wrong s3_location"""
        with self.assertRaises(ValueError) as e:
            _ = EphysJobSettings(
                input_source=CHRONIC_DATA_DIR,
                output_directory=Path("output_dir_s3"),
                compress_job_save_kwargs={"n_jobs": 1},
                s3_location="s2://bucket/session",
            )
        self.assertIn(
            "S3 location must start with 's3://'.",
            str(e.exception),
        )


if __name__ == "__main__":
    unittest.main()
