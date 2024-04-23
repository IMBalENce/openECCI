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

from src import io


def get_relative_stage_pos(reference_image, current_image):
    """
    Compute the relative stage rotation and tilt angles of the current image with respect to the reference image.
    Note that the returned rotation and tilt angles are following the sample reference frame illustrated in Figure 2
    in the Tutorial paper using the right hand rule, such that rotation is around Z\ :sub:`s` axis and tilt is
    around Y\ :sub:`s` axis. For both FEI/Thermofisher and Zeiss microscope, stage rotates clockwisely when a
    positive stage rotation is applied, which is opposite to the convention currently used. Therefore, a negative
    sign should be added when the returned value is used for SEM stage control.

    positive stage rotation => ECP pattern rotates clockwise
    positive stage tilt => ECP pattern moves DOWN

    Supported vendors are:
    * FEI/Thermofisher
    * Zeiss

    Parameters
    ----------
    filename
        Path and file name.

    Returns
        relative stage tilt and rotation in degrees
    -------

    """

    # Read zero reference image to get the initial angles
    ref_metadata = io.get_sem_metadata(reference_image)
    # thermofisher SEM - read stage rot and tilt from image metadata
    cur_metadata = io.get_sem_metadata(current_image)

    # negative sign because the positive SEM rotation direction is opposite to the one used in sample reference frame
    st_rot_angle = -(cur_metadata["stage_rot_angle"] - ref_metadata["stage_rot_angle"])
    st_tilt_angle = cur_metadata["stage_tilt_angle"] - ref_metadata["stage_tilt_angle"]

    diff_x = cur_metadata["stage_x"] - ref_metadata["stage_x"]
    diff_y = cur_metadata["stage_y"] - ref_metadata["stage_y"]
    diff_z = cur_metadata["stage_z"] - ref_metadata["stage_z"]

    if st_rot_angle > 180:
        st_rot_angle = st_rot_angle - 360

    return st_rot_angle, st_tilt_angle, diff_x, diff_y, diff_z


def _stage_in_range(image_path, stage_x, stage_y):
    """
    check if the stage coordinate [stage_x, stage_y] is inside the image

    Returns
        Boolen
    """
    [centre_x, centre_y, hfw, vfh] = _get_image_range(image_path)

    if ((centre_x - hfw / 2) <= stage_x <= (centre_x + hfw / 2)) and (
        (centre_y - vfh / 2) <= stage_y <= (centre_y + vfh / 2)
    ):
        return True
    else:
        return False


def _get_image_range(image_path):
    """
    Get the image centre position and the width, height

    Returns
        [centre_x, centre_y, hfw, vfh]
    """
    metadata = io.get_sem_metadata(image_path)
    centre_x = metadata["stage_x"]
    centre_y = metadata["stage_y"]
    hfw = metadata["pixel_size"] * metadata["resolution"][0]
    vfh = metadata["pixel_size"] * metadata["resolution"][1]

    return [centre_x, centre_y, hfw, vfh]
