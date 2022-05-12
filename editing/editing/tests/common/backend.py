import os
import shutil


def get_directory(backend_name: str,
                  problem:      str,
                  topology:     str) -> str:

    """Return a directory where we can store backend-specific ONNX models."""

    path_package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_zips = os.path.join(path_package, '.backends')           # here we store the ZIP archives contaning backend-specific, QuantLib-exported files
    path_backend_zips = os.path.join(path_zips, backend_name)     # here we store the ZIP archives containing QuantLib-exported ONNX models for a specific backend
    path_topology_zips = os.path.join(path_backend_zips, problem, topology)

    if not os.path.isdir(path_topology_zips):
        os.makedirs(path_topology_zips, exist_ok=True)  # https://stackoverflow.com/a/600612

    return path_topology_zips


def zip_directory(path_dir: str) -> None:
    """Turn a folder of backend-specific files into an archive, then delete it."""
    shutil.make_archive(path_dir, 'zip', path_dir)  # https://stackoverflow.com/a/25650295
    shutil.rmtree(path_dir)
