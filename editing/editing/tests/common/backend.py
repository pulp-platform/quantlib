# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

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
