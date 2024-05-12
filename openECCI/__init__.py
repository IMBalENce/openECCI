# Copyright 2023-2024 The openECCI developers
#
# This file is part of openECCI.
#
# openECCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# openECCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with openECCI. If not, see <http://www.gnu.org/licenses/>.

from openECCI.release import version as __version__

__all__ = [
    "util",
    "convergence_angle",
    "optimization",
    "ctf",
    "rkp",
    "io",
    "stagecomputation",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
