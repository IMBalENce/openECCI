# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

"""
Demo datasets for use when testing functionality.
"""

__all__ = [
    "ebsd_master_pattern",
    "ebsd_map",
    "faraday_cage",
    "fcc_fe",
    "si_wafer",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    _import_mapping = {
        "ebsd_master_pattern": "_data",
        "ebsd_map": "_data",
        "faraday_cage": "_data",
        "fcc_fe": "_data",
        "si_wafer": "_data",
    }
    if name in __all__:
        import importlib

        if name in _import_mapping.keys():
            import_path = f"{__name__}.{_import_mapping.get(name)}"
            return getattr(importlib.import_module(import_path), name)
        else:  # pragma: no cover
            return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
