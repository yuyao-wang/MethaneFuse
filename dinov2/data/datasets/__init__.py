# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

__all__ = ["DummyDataset", "FmowDataset"]


def __getattr__(name):
    if name == "DummyDataset":
        from .dummy import DummyDataset

        return DummyDataset
    if name == "FmowDataset":
        from .fmow import FmowDataset

        return FmowDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
