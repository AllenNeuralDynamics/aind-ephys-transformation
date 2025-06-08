"""Compression utilities for writing recordings to Zarr format."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr

from spikeinterface import BaseRecording
from spikeinterface.core.core_tools import check_json
from spikeinterface.core.job_tools import split_job_kwargs
from spikeinterface.core.zarrextractors import (
    get_default_zarr_compressor,
    add_properties_and_annotations,
)


def write_or_append_recording_to_zarr(
    recording: BaseRecording,
    folder_path: str | Path,
    storage_options: dict | None = None,
    annotations_to_update: dict | None = None,
    **kwargs,
):
    """Write or append a recording to a Zarr folder."""
    zarr_root = zarr.open(
        str(folder_path), mode="a", storage_options=storage_options
    )
    if recording.get_num_segments() > 1:
        raise ValueError(
            "write_or_append_recording_to_zarr does not support "
            "multi-segment recordings."
        )
    global_start_frame = recording.get_annotation(
        "sample_index_from_session_start"
    )
    if global_start_frame is None:
        global_start_frame = 0
    add_or_append_recording_to_zarr_group(
        recording,
        zarr_root,
        global_start_frame=global_start_frame,
        annotations_to_update=annotations_to_update,
        **kwargs,
    )


def add_or_append_recording_to_zarr_group(  # noqa: C901
    recording: BaseRecording,
    zarr_group: zarr.hierarchy.Group,
    verbose=False,
    dtype=None,
    global_start_frame=0,
    annotations_to_update=None,
    **kwargs,
):
    """Add or append a recording to a Zarr group."""
    zarr_kwargs, job_kwargs = split_job_kwargs(kwargs)

    # we don't write the provenance since the recording keeps growing
    zarr_group.attrs["provenance"] = None

    # save data (done the subclass)
    if "sampling_frequency" not in zarr_group.attrs:
        zarr_group.attrs["sampling_frequency"] = float(
            recording.get_sampling_frequency()
        )
    if "num_channels" not in zarr_group.attrs:
        zarr_group.attrs["num_segments"] = 1
    dataset_paths = [f"traces_seg{0}"]

    dtype = recording.get_dtype() if dtype is None else dtype
    channel_chunk_size = zarr_kwargs.get("channel_chunk_size", None)
    global_compressor = zarr_kwargs.pop(
        "compressor", get_default_zarr_compressor()
    )
    compressor_by_dataset = zarr_kwargs.pop("compressor_by_dataset", {})
    global_filters = zarr_kwargs.pop("filters", None)
    filters_by_dataset = zarr_kwargs.pop("filters_by_dataset", {})

    if "channel_ids" not in zarr_group:
        zarr_group.create_dataset(
            name="channel_ids",
            data=recording.get_channel_ids(),
            compressor=None,
        )

    compressor_traces = compressor_by_dataset.get("traces", global_compressor)
    filters_traces = filters_by_dataset.get("traces", global_filters)
    add_or_append_traces_to_zarr(
        recording=recording,
        zarr_group=zarr_group,
        dataset_paths=dataset_paths,
        compressor=compressor_traces,
        filters=filters_traces,
        dtype=dtype,
        channel_chunk_size=channel_chunk_size,
        global_start_frame=global_start_frame,
        verbose=verbose,
        **job_kwargs,
    )

    # save probe
    if "contact_vector" not in zarr_group:
        if recording.get_property("contact_vector") is not None:
            probegroup = recording.get_probegroup()
            zarr_group.attrs["probe"] = check_json(
                probegroup.to_dict(array_as_list=True)
            )

    # save time vector if any
    t_starts = np.zeros(recording.get_num_segments(), dtype="float64") * np.nan
    for segment_index, rs in enumerate(recording._recording_segments):
        d = rs.get_times_kwargs()
        time_vector = d["time_vector"]

        compressor_times = compressor_by_dataset.get(
            "times", global_compressor
        )
        filters_times = filters_by_dataset.get("times", global_filters)

        if time_vector is not None:
            time_dset_name = f"times_seg{segment_index}"
            if time_dset_name not in zarr_group:
                _ = zarr_group.create_dataset(
                    name=f"times_seg{segment_index}",
                    data=time_vector,
                    filters=filters_times,
                    compressor=compressor_times,
                )
            else:
                time_dset = zarr_group[time_dset_name]
                if len(time_dset) < global_start_frame + len(time_vector):
                    # Resize the dataset if it is smaller than expected
                    time_dset.resize((global_start_frame + len(time_vector),))
                global_end_frame = global_start_frame + len(time_vector)
                time_dset[global_start_frame:global_end_frame] = time_vector

        elif d["t_start"] is not None:
            t_starts[segment_index] = d["t_start"]

    if np.any(~np.isnan(t_starts)) and "t_starts" not in zarr_group:
        zarr_group.create_dataset(
            name="t_starts", data=t_starts, compressor=None
        )

    if "properties" not in zarr_group:
        add_properties_and_annotations(zarr_group, recording)

    if annotations_to_update is not None:
        for key in annotations_to_update:
            value = recording.get_annotation(key)
            if key in zarr_group.attrs:
                new_value = zarr_group.attrs[key]
                new_value.update(value)
                value = new_value
            zarr_group.attrs[key] = value


def add_or_append_traces_to_zarr(
    recording,
    zarr_group,
    dataset_paths,
    channel_chunk_size=None,
    dtype=None,
    compressor=None,
    filters=None,
    global_start_frame=0,
    verbose=False,
    **job_kwargs,
):
    """
    Save the trace of a recording extractor in several zarr format.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object to be saved in .dat format
    zarr_group : zarr.Group
        The zarr group to add traces to
    dataset_paths : list
        List of paths to traces datasets in the zarr group
    channel_chunk_size : int or None, default: None (chunking in time only)
        Channels per chunk
    dtype : dtype, default: None
        Type of the saved data
    compressor : zarr compressor or None, default: None
        Zarr compressor
    filters : list, default: None
        List of zarr filters
    global_start_frame : int, default: 0
        The global start frame to use for appending traces.
    verbose : bool, default: False
        If True, output is verbose (when chunks are used)
    {}
    """
    from spikeinterface.core.job_tools import (
        ensure_chunk_size,
        fix_job_kwargs,
        ChunkRecordingExecutor,
    )

    assert dataset_paths is not None, "Provide 'file_path'"

    dtype = recording.get_dtype()

    job_kwargs = fix_job_kwargs(job_kwargs)
    chunk_size = ensure_chunk_size(recording, **job_kwargs)

    # create zarr datasets files
    zarr_datasets = []
    # only 1 segment supported
    segment_index = 0
    num_frames = recording.get_num_samples(segment_index)
    num_channels = recording.get_num_channels()
    dset_name = dataset_paths[segment_index]
    shape = (num_frames, num_channels)
    if dset_name in zarr_group:
        dset = zarr_group[dset_name]
        if dset.shape[0] < global_start_frame + num_frames:
            dset.resize((global_start_frame + num_frames, num_channels))
    else:
        dset = zarr_group.create_dataset(
            name=dset_name,
            shape=shape,
            chunks=(chunk_size, channel_chunk_size),
            dtype=dtype,
            filters=filters,
            compressor=compressor,
        )
    zarr_datasets.append(dset)

    # use executor (loop or workers)
    func = _write_zarr_chunk_append
    init_func = _init_zarr_worker_append
    init_args = (recording, zarr_datasets, dtype, global_start_frame)
    executor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        verbose=verbose,
        job_name="write_zarr_recording",
        **job_kwargs,
    )
    recording_slices = get_recording_slices_aligned_to_zarr_chunks(
        recording, chunk_size, global_start_frame
    )
    executor.run(recording_slices=recording_slices)


def get_recording_slices_aligned_to_zarr_chunks(
    recording: BaseRecording, chunk_size: int, global_start_frame: int = 0
):
    """
    This function returns a list of tuples representing slices of the recording
    that are aligned to Zarr chunks.
    Normally, each job writes to a different chunk, except for the last one
    which may be smaller than the chunk size. When appending, we need to make
    sure that the first appended chunk "completes" the previous one, so that
    additional chunks will be aligned to the chunk size.

    Parameters
    ----------
    recording : BaseRecording
        The recording extractor object.
    chunk_size : int
        The size of the chunks in frames.
    global_start_frame : int, default: 0
        The global start frame to use for appending traces.

    Returns
    -------
    recording_slices : list of tuples
        A list of tuples where each tuple represents a segment of the recording
        aligned to Zarr chunks.
    """
    from spikeinterface.core.job_tools import divide_segment_into_chunks

    segment_index = 0
    first_chunk_size = chunk_size - global_start_frame % chunk_size
    recording_slices = [(segment_index, 0, first_chunk_size)]
    num_frames = recording.get_num_samples(segment_index) - first_chunk_size
    chunks = divide_segment_into_chunks(num_frames, chunk_size)
    recording_slices.extend(
        [
            (
                segment_index,
                frame_start + first_chunk_size,
                frame_stop + first_chunk_size,
            )
            for frame_start, frame_stop in chunks
        ]
    )
    return recording_slices


# used by write_zarr_recording + ChunkRecordingExecutor
def _init_zarr_worker_append(
    recording, zarr_datasets, dtype, global_start_frame
):
    """Initialize the worker context for appending to Zarr."""

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["zarr_datasets"] = zarr_datasets
    worker_ctx["dtype"] = np.dtype(dtype)
    worker_ctx["global_start_frame"] = global_start_frame

    return worker_ctx


# used by write_zarr_recording + ChunkRecordingExecutor
def _write_zarr_chunk_append(
    segment_index, start_frame, end_frame, worker_ctx
):
    """Write a chunk of traces to Zarr for appending."""
    import gc

    # recover variables of the worker
    recording = worker_ctx["recording"]
    dtype = worker_ctx["dtype"]
    zarr_dataset = worker_ctx["zarr_datasets"][segment_index]
    global_start_frame = worker_ctx["global_start_frame"]

    # apply function
    traces = recording.get_traces(
        start_frame=start_frame,
        end_frame=end_frame,
        segment_index=segment_index,
    )
    traces = traces.astype(dtype)
    start_frame += global_start_frame
    end_frame += global_start_frame
    zarr_dataset[start_frame:end_frame, :] = traces

    # fix memory leak by forcing garbage collection
    del traces
    gc.collect()
