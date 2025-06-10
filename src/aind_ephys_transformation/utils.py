"""
Utility functions for image readers
"""
import platform
import subprocess


from aind_ephys_transformation.models import PathLike


def sync_dir_to_s3(  # pragma: no cover
    directory_to_upload: PathLike, s3_location: str, exclude=None
) -> None:
    """
    Syncs a local directory to an s3 location by running aws cli in a
    subprocess.

    Parameters
    ----------
    directory_to_upload : PathLike
    s3_location : str

    Returns
    -------
    None

    """
    # Upload to s3
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    base_command = [
        "aws",
        "s3",
        "sync",
        str(directory_to_upload),
        s3_location,
        "--only-show-errors",
    ]
    if exclude is not None:
        for exclude_pattern in exclude:
            base_command.extend(["--exclude", exclude_pattern])

    subprocess.run(base_command, shell=shell, check=True)


def copy_file_to_s3(  # pragma: no cover
    file_to_upload: PathLike,
    s3_location: str
) -> None:
    """
    Syncs a local directory to an s3 location by running aws cli in a
    subprocess.

    Parameters
    ----------
    file_to_upload : PathLike
    s3_location : str

    Returns
    -------
    None

    """
    # Upload to s3
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    base_command = [
        "aws",
        "s3",
        "cp",
        str(file_to_upload),
        s3_location,
        "--only-show-errors",
    ]

    subprocess.run(base_command, shell=shell, check=True)
