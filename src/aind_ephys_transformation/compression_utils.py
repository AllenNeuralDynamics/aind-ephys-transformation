"""Compression utilities for writing recordings to Zarr format."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr

from spikeinterface import BaseRecording
from spikeinterface.core.core_tools import check_json
from spikeinterface.core.job_tools import split_job_kwargs
from spikeinterface.core.zarr_tools import (
    build_codec_pipeline,
    adjust_chunks_shards_and_job_kwargs,
)
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
    zarr.consolidate_metadata(zarr_root.store)


def add_or_append_recording_to_zarr_group(  # noqa: C901
    recording: BaseRecording,
    zarr_group: zarr.Group,
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
    dataset_paths = ["traces_seg0"]
    if recording.has_time_vector():
        dataset_timestamps_paths = ["times_seg0"]
    else:
        dataset_timestamps_paths = None

    dtype = recording.get_dtype() if dtype is None else dtype
    channel_chunk_size = zarr_kwargs.get("channel_chunk_size", None)
    shards = zarr_kwargs.get("shards", None)
    shard_factor = zarr_kwargs.get("shard_factor", None)
    buffer_megabytes = zarr_kwargs.get("buffer_megabytes", 500)
    global_compressor = zarr_kwargs.pop(
        "compressor", get_default_zarr_compressor()
    )
    compressor_by_dataset = zarr_kwargs.pop("compressor_by_dataset", {})
    global_filters = zarr_kwargs.pop("filters", None)
    filters_by_dataset = zarr_kwargs.pop("filters_by_dataset", {})

    if "channel_ids" not in zarr_group:
        # variable-length UTF-8 is the stable zarr v3 spec for strings
        channel_ids = recording.get_channel_ids()
        if channel_ids.dtype.kind in ("U", "S"):
            channel_ids = channel_ids.astype("T")
        zarr_group.create_array(
            name="channel_ids",
            data=channel_ids,
            compressors=None,
        )

    compressor_traces = compressor_by_dataset.get("traces", global_compressor)
    filters_traces = filters_by_dataset.get("traces", global_filters)
    compressor_times = compressor_by_dataset.get("times", global_compressor)
    filters_times = filters_by_dataset.get("times", global_filters)
    add_or_append_traces_to_zarr(
        recording=recording,
        zarr_group=zarr_group,
        dataset_paths=dataset_paths,
        dataset_timestamps_paths=dataset_timestamps_paths,
        compressor_traces=compressor_traces,
        filters_traces=filters_traces,
        compressor_times=compressor_times,
        filters_times=filters_times,
        dtype=dtype,
        channel_chunk_size=channel_chunk_size,
        shards=shards,
        shard_factor=shard_factor,
        buffer_megabytes=buffer_megabytes,
        global_start_frame=global_start_frame,
        verbose=verbose,
        **job_kwargs,
    )

    # save probe
    if recording.has_probe():
        probegroup = recording.get_probegroup()
        zarr_group.attrs["probegroup"] = check_json(
            probegroup.to_dict(array_as_list=True)
        )

    # save time vector if any
    t_starts = np.zeros(recording.get_num_segments(), dtype="float64") * np.nan
    for segment_index, rs in enumerate(recording._recording_segments):
        d = rs.get_times_kwargs()
        if d["t_start"] is not None:
            t_starts[segment_index] = d["t_start"]

    if np.any(~np.isnan(t_starts)) and "t_starts" not in zarr_group:
        zarr_group.create_array(
            name="t_starts", data=t_starts, compressors=None
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
    dataset_timestamps_paths=None,
    channel_chunk_size=None,
    shards=None,
    shard_factor=None,
    buffer_megabytes=500,
    dtype=None,
    compressor_traces=None,
    filters_traces=None,
    compressor_times=None,
    filters_times=None,
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
    shards : tuple or None, default: None
        Shard shape for the traces dataset (ignored when appending)
    shard_factor : int or None, default: None
        Number of chunks per shard in time (ignored when appending)
    buffer_megabytes : int, default: 500
        Buffer size in megabytes for writing data to zarr (per worker)
    dtype : dtype, default: None
        Type of the saved data
    compressor_traces : zarr compressor or None, default: None
        Zarr compressor for traces
    filters_traces : list, default: None
        List of zarr filters for traces
    compressor_times : zarr compressor or None, default: None
        Zarr compressor for times
    filters_times : list, default: None
        List of zarr filters for times
    global_start_frame : int, default: 0
        The global start frame to use for appending traces.
    verbose : bool, default: False
        If True, output is verbose (when chunks are used)
    {}
    """
    from spikeinterface.core.job_tools import (
        fix_job_kwargs,
        TimeSeriesChunkExecutor,
    )

    assert dataset_paths is not None, "Provide 'file_path'"

    dtype = recording.get_dtype() if dtype is None else dtype

    job_kwargs = fix_job_kwargs(job_kwargs)

    # only 1 segment supported
    segment_index = 0

    # create zarr datasets files
    zarr_datasets = []
    num_frames = recording.get_num_samples(segment_index)
    num_channels = recording.get_num_channels()
    dset_name = dataset_paths[segment_index]
    shape = (global_start_frame + num_frames, num_channels)
    if dset_name in zarr_group:
        dset = zarr_group[dset_name]
        # when appending, the on-disk chunks/shards dictate the job layout
        chunks, shards, job_kwargs = adjust_chunks_shards_and_job_kwargs(
            chunks=dset.chunks,
            shards=dset.shards,
            job_kwargs=job_kwargs,
            time_series=recording,
        )
        if dset.shape[0] < global_start_frame + num_frames:
            dset.resize((global_start_frame + num_frames, num_channels))
    else:
        chunks, shards, job_kwargs = adjust_chunks_shards_and_job_kwargs(
            extra_chunks=(
                (channel_chunk_size,)
                if channel_chunk_size is not None
                else None
            ),
            shards=shards,
            shard_factor=shard_factor,
            job_kwargs=job_kwargs,
            time_series=recording,
        )
        codec_kwargs = build_codec_pipeline(filters_traces, compressor_traces)
        dset = zarr_group.create_array(
            name=dset_name,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            **codec_kwargs,
        )
    zarr_datasets.append(dset)

    zarr_timestamps_datasets = []
    if dataset_timestamps_paths is not None:
        timestamps_dset_name = dataset_timestamps_paths[segment_index]
        if timestamps_dset_name in zarr_group:
            timestamps_dset = zarr_group[timestamps_dset_name]
            if timestamps_dset.shape[0] < global_start_frame + num_frames:
                timestamps_dset.resize((global_start_frame + num_frames,))
        else:
            codec_kwargs = build_codec_pipeline(
                filters_times, compressor_times
            )
            timestamps_dset = zarr_group.create_array(
                name=timestamps_dset_name,
                shape=(global_start_frame + num_frames,),
                chunks=(chunks[0],),
                shards=(shards[0],) if shards is not None else None,
                dtype="float64",
                **codec_kwargs,
            )
        zarr_timestamps_datasets.append(timestamps_dset)

    # use executor (loop or workers)
    func = _write_zarr_chunk_append
    init_func = _init_zarr_worker_append
    init_args = (
        recording,
        zarr_datasets,
        zarr_timestamps_datasets,
        dtype,
        global_start_frame,
        buffer_megabytes,
    )
    executor = TimeSeriesChunkExecutor(
        recording,
        func,
        init_func,
        init_args,
        verbose=verbose,
        job_name="write_zarr_recording",
        **job_kwargs,
    )
    # align jobs to shards (the unit of write) so no two jobs share a shard
    slices = get_slices_aligned_to_zarr_chunks(
        recording,
        shards[0] if shards is not None else chunks[0],
        global_start_frame,
    )
    executor.run(slices=slices)


def get_slices_aligned_to_zarr_chunks(
    recording: BaseRecording,
    chunk_size: int,
    global_start_frame: int = 0,
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
    slices : list of tuples
        A list of tuples where each tuple represents a segment of the recording
        aligned to Zarr chunks.
    """
    from spikeinterface.core.job_tools import divide_segment_into_chunks

    segment_index = 0
    num_samples = recording.get_num_samples(segment_index)
    first_chunk_size = min(
        chunk_size - global_start_frame % chunk_size, num_samples
    )
    slices = [(segment_index, 0, first_chunk_size)]
    num_frames = num_samples - first_chunk_size
    chunks = divide_segment_into_chunks(num_frames, chunk_size)
    slices.extend(
        [
            (
                segment_index,
                frame_start + first_chunk_size,
                frame_stop + first_chunk_size,
            )
            for frame_start, frame_stop in chunks
        ]
    )
    return slices


# used by write_zarr_recording + TimeSeriesChunkExecutor
def _init_zarr_worker_append(
    recording,
    zarr_datasets,
    zarr_timestamps_datasets,
    dtype,
    global_start_frame,
    buffer_megabytes,
):
    """Initialize the worker context for appending to Zarr."""

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["zarr_datasets"] = zarr_datasets
    worker_ctx["zarr_timestamps_datasets"] = zarr_timestamps_datasets
    worker_ctx["dtype"] = np.dtype(dtype)
    worker_ctx["global_start_frame"] = global_start_frame
    worker_ctx["buffer_megabytes"] = buffer_megabytes

    return worker_ctx


# used by write_zarr_recording + TimeSeriesChunkExecutor
def _write_zarr_chunk_append(
    segment_index, start_frame, end_frame, worker_ctx
):
    """Write a chunk of traces to Zarr for appending."""
    import gc

    # recover variables of the worker
    recording = worker_ctx["recording"]
    dtype = worker_ctx["dtype"]
    zarr_dataset = worker_ctx["zarr_datasets"][segment_index]
    zarr_timestamps_datasets = worker_ctx["zarr_timestamps_datasets"]

    global_start_frame = worker_ctx["global_start_frame"]
    start_frame_global = start_frame + global_start_frame
    end_frame_global = end_frame + global_start_frame

    # With shards, a job covers a whole shard so only one worker writes to it.
    # Traces are computed one zarr chunk at a time to keep the memory of the
    # chain small, and written as many whole chunks at a time as fit in
    # buffer_megabytes, since zarr reads and rewrites the whole shard on every
    # write. Boundaries are aligned to chunks in the global (on-disk) frame.
    chunk_size = zarr_dataset.chunks[0]
    bytes_per_chunk = (
        chunk_size * int(np.prod(zarr_dataset.shape[1:])) * dtype.itemsize
    )
    write_size = chunk_size * max(
        1, worker_ctx["buffer_megabytes"] * 1_000_000 // bytes_per_chunk
    )
    write_start = start_frame_global
    while write_start < end_frame_global:
        write_end = min(
            (write_start // write_size + 1) * write_size, end_frame_global
        )
        traces = np.empty(
            (write_end - write_start,) + zarr_dataset.shape[1:], dtype=dtype
        )
        read_start = write_start
        while read_start < write_end:
            read_end = min(
                (read_start // chunk_size + 1) * chunk_size, write_end
            )
            traces[read_start - write_start:read_end - write_start] = (
                recording.get_traces(
                    start_frame=read_start - global_start_frame,
                    end_frame=read_end - global_start_frame,
                    segment_index=segment_index,
                )
            )
            read_start = read_end
        zarr_dataset[write_start:write_end, :] = traces
        write_start = write_end

    if len(zarr_timestamps_datasets) > 0:
        zarr_timestamps_dataset = zarr_timestamps_datasets[segment_index]
        timestamps = recording.get_times()[start_frame:end_frame]
        zarr_timestamps_dataset[start_frame_global:end_frame_global] = (
            timestamps
        )
        del timestamps

    # fix memory leak by forcing garbage collection
    gc.collect()
