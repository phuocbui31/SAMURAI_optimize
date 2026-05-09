# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
except ModuleNotFoundError:
    # Allow importing lightweight submodules in environments that only run
    # smoke tests. Model construction still requires hydra via build_sam.py.
    pass
else:
    if not GlobalHydra.instance().is_initialized():
        initialize_config_module("sam2", version_base="1.2")
