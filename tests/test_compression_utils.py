"""Tests for compression utilities in the Ephys Transformation pipeline."""

import unittest
import shutil
from pathlib import Path
import numpy as np
import zarr

from spikeinterface import generate_ground_truth_recording

from aind_ephys_transformation.compression_utils import (
    write_or_append_recording_to_zarr,
)


class TestEphysJob(unittest.TestCase):
    """Tests for compression_utils module"""

    @classmethod
    def setUpClass(cls):
        """Setup basic job settings and job that can be used across tests"""
        cls.recording1, _ = generate_ground_truth_recording(
            durations=[10], seed=2308
        )
        cls.recording2, _ = generate_ground_truth_recording(
            durations=[20], seed=2205
        )
        cls.recording_multi_segment1, _ = (
            generate_ground_truth_recording(
                durations=[10, 5], seed=2308
            )
        )
        cls.recording_multi_segment2, _ = (
            generate_ground_truth_recording(
                durations=[20, 15], seed=2308
            )
        )
        cls.tmp_path = Path("tmp_test_ephys_compression_utils")

    @classmethod
    def tearDownClass(cls):
        """Delete temporary directory if it exists."""
        tmp_path = cls.tmp_path
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

    def test_write_new_recording(self):
        """Test writing a new recording to a zarr file."""
        temp_zarr_path = self.tmp_path / "test.zarr"
        write_or_append_recording_to_zarr(
            self.recording1, temp_zarr_path, n_jobs=2
        )

        # Verify the data was written correctly
        zarr_data = zarr.open(str(temp_zarr_path), mode="r")
        np.testing.assert_array_equal(
            zarr_data["traces_seg0"][:], self.recording1.get_traces()
        )

    def test_append_to_existing_recording(self):
        """Test appending to an existing zarr file."""
        # First write
        temp_zarr_path = self.tmp_path / "test2.zarr"

        write_or_append_recording_to_zarr(
            self.recording1, temp_zarr_path, n_jobs=2
        )

        # Append the same data
        write_or_append_recording_to_zarr(
            self.recording2, temp_zarr_path, n_jobs=2
        )

        # Verify the data was appended correctly
        zarr_data = zarr.open(str(temp_zarr_path), mode="r")
        expected_shape = (
            self.recording1.get_num_samples()
            + self.recording2.get_num_samples(),
            self.recording1.get_num_channels(),
        )
        assert zarr_data["traces_seg0"].shape == expected_shape

        # First half should match original data
        np.testing.assert_array_equal(
            zarr_data["traces_seg0"][
                : self.recording1.get_num_samples()
            ],
            self.recording1.get_traces(),
        )
        # Second half should match appended data
        np.testing.assert_array_equal(
            zarr_data["traces_seg0"][self.recording1.get_num_samples():],
            self.recording2.get_traces(),
        )

    def test_write_multiple_segments(self):
        """Test writing multiple segments."""
        # Create recording with two segments
        temp_zarr_path = self.tmp_path / "multi_segment.zarr"

        # Write first recording
        write_or_append_recording_to_zarr(
            self.recording_multi_segment1, temp_zarr_path, n_jobs=2
        )

        zarr_data = zarr.open(str(temp_zarr_path), mode="r")

        for segment_index in range(
            self.recording_multi_segment1.get_num_segments()
        ):
            # Verify the data was written correctly
            np.testing.assert_array_equal(
                zarr_data[f"traces_seg{segment_index}"][:],
                self.recording_multi_segment1.get_traces(
                    segment_index=segment_index
                ),
            )

    def test_append_multiple_segments(self):
        """Test appending multiple segments to an existing zarr file."""
        temp_zarr_path = self.tmp_path / "multi_segment2.zarr"

        # Write first recording
        write_or_append_recording_to_zarr(
            self.recording_multi_segment1,
            temp_zarr_path,
        )
        # Append second recording
        write_or_append_recording_to_zarr(
            self.recording_multi_segment2, temp_zarr_path
        )

        # Verify both segments
        zarr_data = zarr.open(str(temp_zarr_path), mode="r")
        for segment_index in range(
            self.recording_multi_segment1.get_num_segments()
        ):
            segment_traces1 = (
                self.recording_multi_segment1.get_traces(
                    segment_index
                )
            )
            segment_traces2 = (
                self.recording_multi_segment2.get_traces(
                    segment_index
                )
            )
            segment_name = f"traces_seg{segment_index}"

            num_samples1 = (
                self.recording_multi_segment1.get_num_samples(
                    segment_index
                )
            )

            # first recording
            if segment_index == 0:
                np.testing.assert_array_equal(
                    zarr_data[segment_name][:num_samples1],
                    segment_traces1,
                )
            else:
                np.testing.assert_array_equal(
                    zarr_data[segment_name][num_samples1:],
                    segment_traces2,
                )

    def test_append_with_times(self):
        """Test appending recordings with time vectors."""
        # Create a temporary zarr file
        temp_zarr_path = self.tmp_path / "test_times.zarr"

        # Write first recording with time vector
        self.recording1.set_times(self.recording1.get_times() + 100)
        write_or_append_recording_to_zarr(
            self.recording1, temp_zarr_path, n_jobs=2
        )

        # Append second recording with time vector
        self.recording2.set_times(self.recording2.get_times() + 300)
        write_or_append_recording_to_zarr(
            self.recording2, temp_zarr_path, n_jobs=2
        )

        # Verify the time vectors were appended correctly
        zarr_data = zarr.open(temp_zarr_path, mode="r")
        np.testing.assert_array_equal(
            zarr_data["times_seg0"][:],
            np.concatenate(
                [
                    self.recording1.get_times(),
                    self.recording2.get_times(),
                ]
            ),
        )

    def test_write_with_t_start(self):
        """Test writing a recording with t_start."""
        temp_zarr_path = self.tmp_path / "test_t_start.zarr"

        # Set t_start for the recording
        self.recording1._recording_segments[0].t_start = 0
        self.recording1.shift_times(50.0)

        # Write the recording
        write_or_append_recording_to_zarr(
            self.recording1, temp_zarr_path, n_jobs=2
        )

        # Verify t_start is saved correctly
        zarr_data = zarr.open(temp_zarr_path, mode="r")
        np.testing.assert_array_equal(
            zarr_data["t_starts"], np.array([50.0])
        )

    def test_with_provenance(self):
        """Test writing a recording with provenance."""
        temp_zarr_path = self.tmp_path / "test_provenance.zarr"

        # Set provenance for the recording
        recording2 = self.recording1.save(
            folder=self.tmp_path / "cached"
        )

        # Write the recording
        write_or_append_recording_to_zarr(
            recording2, temp_zarr_path, n_jobs=2
        )

        # Verify provenance is saved correctly
        zarr_data = zarr.open(temp_zarr_path, mode="r")
        zarr_data.attrs["provenance"] is not None
