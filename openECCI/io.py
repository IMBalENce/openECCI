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

from tifffile import TiffFile, imread
import numpy as np
import math
import re
import h5py as h5
import xml.etree.ElementTree as ET
import pathlib
from orix import plot, crystal_map
from openECCI.ctf import file_reader as ctf_reader
from orix.io import load as ang_reader
from orix.quaternion import Rotation
import pickle


def get_sem_metadata(filename: str) -> dict:
    """
    Retrieve relevant SEM metadate, e.g. stage rotation/tilt angle, image resolution, SEM vendor
    and whether image databar is included in the image. For FEI/Thermofisher SEM, databar is
    additional pixels add at the bottom of image pixels; For Zeiss, databar is added as image
    overlays on top of image pixels.

    Supported vendors are:
    * FEI/Thermofisher
    * Zeiss

    Parameters
    ----------
    filename
        Path and file name from SEM.

    Returns
    -------
    sem_metadata
        Metadata dictionary.
        {"vendor": vendor,
        "stage_rot_angle":st_rot_angle,
        "stage_tilt_angle":st_tilt_angle,
        "stage_x": st_x,
        "stage_y": st_y,
        "stage_z": st_z,
        "resolution": resolution,
        "pixel_size": pixel_size,
        "databar":databar,
        "HV": HV,
        "WD": WD,
        "beam_current": beam_current,
        "aperture_diameter": aperture_diameter,
        "column_mode": column_mode,
        }

    """
    file_type = get_file_footprint(filename)

    if file_type in ["FEI_SEM", "Zeiss_SEM"]:
        image_dim = imread(filename).shape

        with TiffFile(filename) as tif:
            if tif.fei_metadata:
                """
                FEI/Thermofisher SEM metadata
                """
                st_rot_angle = math.degrees(tif.fei_metadata["Stage"]["StageR"])
                st_tilt_angle = math.degrees(tif.fei_metadata["Stage"]["StageT"])
                # st_x = tif.fei_metadata['Stage']['StageX']
                # st_y = tif.fei_metadata['Stage']['StageY']
                st_z = tif.fei_metadata["Stage"]["StageZ"]

                # get the true image position including the stage position and the beam shift
                st_x = (
                    tif.fei_metadata["Stage"]["StageX"]
                    - tif.fei_metadata["Beam"]["BeamShiftX"]
                )
                st_y = (
                    tif.fei_metadata["Stage"]["StageY"]
                    - tif.fei_metadata["Beam"]["BeamShiftY"]
                )

                pixel_width = tif.fei_metadata["EScan"]["PixelWidth"]
                pixel_height = tif.fei_metadata["EScan"]["PixelHeight"]
                WD = tif.fei_metadata["EBeam"]["WD"]
                if pixel_width == pixel_height:
                    pixel_size = pixel_width
                else:
                    pixel_size = None
                    Warning("Image pixel is not square!")

                resolution = [
                    tif.fei_metadata["Image"]["ResolutionX"],
                    tif.fei_metadata["Image"]["ResolutionY"],
                ]
                beam_current = tif.fei_metadata["EBeam"]["BeamCurrent"]
                aperture_diameter = tif.fei_metadata["EBeam"]["ApertureDiameter"]
                column_mode = tif.fei_metadata["EBeam"]["LensMode"]

                vendor = "FEI"
                HV = tif.fei_metadata["EBeam"]["HV"] / 1000
            elif tif.sem_metadata["dp_sem"]:
                """
                Zeiss SEM metadata
                """
                length_unit_lookup = {"nm": 1e9, "µm": 1e6, "mm": 1e3}
                current_unit_lookup = {"nA": 1e9, "pA": 1e12}

                st_rot_angle = tif.sem_metadata["ap_stage_at_r"][1]
                st_tilt_angle = tif.sem_metadata["ap_stage_at_t"][1]
                st_x = (
                    tif.sem_metadata["ap_stage_at_x"][1]
                    * length_unit_lookup[tif.sem_metadata["ap_stage_at_x"][2]]
                )  # in m
                st_y = (
                    tif.sem_metadata["ap_stage_at_y"][1]
                    * length_unit_lookup[tif.sem_metadata["ap_stage_at_y"][2]]
                )  # in m
                st_z = (
                    tif.sem_metadata["ap_stage_at_z"][1]
                    * length_unit_lookup[tif.sem_metadata["ap_stage_at_z"][2]]
                )  # in m

                pixel_size_unit = tif.sem_metadata["ap_pixel_size"][2]
                current_unit = tif.sem_metadata["ap_iprobe"][2]
                aperture_dia_unit = tif.sem_metadata["ap_aperturesize"][2]
                WD_unit = tif.sem_metadata["ap_wd"][2]

                WD = tif.sem_metadata["ap_wd"][1] * length_unit_lookup[WD_unit]  # in m
                pixel_size = (
                    tif.sem_metadata["ap_pixel_size"][1]
                    * length_unit_lookup[pixel_size_unit]
                )  # in m
                resolution = [
                    int(i) for i in tif.sem_metadata["dp_image_store"][1].split(" * ")
                ]
                beam_current = (
                    tif.sem_metadata["ap_iprobe"][1] * current_unit_lookup[current_unit]
                )  # in A
                aperture_diameter = (
                    tif.sem_metadata["ap_aperturesize"][1]
                    * length_unit_lookup[aperture_dia_unit]
                )  # in m
                column_mode = tif.sem_metadata["dp_column_mode"][1]

                HV = tif.sem_metadata["ap_actualkv"][1]
                vendor = "Zeiss"

        if resolution[1] < image_dim[0]:
            databar = True
        else:
            databar = False

        sem_metadata = {
            "vendor": vendor,
            "stage_rot_angle": st_rot_angle,
            "stage_tilt_angle": st_tilt_angle,
            "stage_x": st_x,
            "stage_y": st_y,
            "stage_z": st_z,
            "resolution": resolution,
            "pixel_size": pixel_size,
            "databar": databar,
            "HV": HV,
            "WD": WD,
            "beam_current": beam_current,
            "aperture_diameter": aperture_diameter,
            "column_mode": column_mode,
        }
    else:
        sem_metadata = None
        raise IOError(f"{filename} is not supported format")

    return sem_metadata


def get_file_footprint(filename: str) -> str:
    """
    Retrieve the file type of the input file.

    Supported formats are:
    * FEI/Thermofisher SEM .tiff file:          "FEI_SEM"
    * Zeiss SEM .tiff file:                     "Zeiss_SEM"
    * Oxford/HKL .ctf file:                     "hkl_ctf"
    * Oxford/HKL raw pattern .tiff file:        "oina-image"
    * Oxford/HKL .hoina file:                   "hkl_hdf5"
    * EDAX .ang file                            "tsl_ang"
    * EDAX .hdf5 master pattern                 "tsl_hdf5"
    * EMsoft .hdf5 file                         "emsoft_hdf5"
    * Kikuchipy .hdf5 file                      "kpy_hdf5"

    Parameters
    ----------
    filename
        Path and file name.

    Returns
    -------
    file_type
        File type discription in str.
    """
    file_type = None
    file_extension = pathlib.Path(filename).suffix
    if file_extension in [".tif", ".tiff", ".TIFF"]:
        try:
            ebsp_metadata = get_hkl_metadata(
                filename=filename,
            )
            if ebsp_metadata["image_type"] == "oina-image":
                file_type = ebsp_metadata["image_type"]
        except KeyError:
            try:
                with TiffFile(filename) as tif:
                    if tif.fei_metadata is not None:
                        file_type = "FEI_SEM"
                    elif tif.sem_metadata["dp_sem"] is not None:
                        file_type = "Zeiss_SEM"
                    else:
                        raise KeyError
            except KeyError:
                raise IOError("{filename} is not supported format")

    elif file_extension in ".ctf":
        file_type = "hkl_ctf"
    elif file_extension in ".ang":
        file_type = "tsl_ang"
    elif file_extension in [".h5", ".hdf5", ".h5oina"]:
        with h5.File(filename, "r") as f:
            try:
                manufacturer = f["Manufacturer"][:][0].decode()
                if manufacturer != "Oxford Instruments":
                    raise KeyError
                else:
                    file_type = "hkl_hdf5"
            except KeyError:
                try:
                    manufacturer = f[" Manufacturer"][:][0].decode()
                    if manufacturer != "EDAX":
                        raise KeyError
                    else:
                        file_type = "tsl_hdf5"
                except KeyError:
                    try:
                        manufacturer = f["EMheader/EBSDmaster/ProgramName"][:][
                            0
                        ].decode()
                        if manufacturer not in ["EMEBSDmaster.f90"]:
                            raise KeyError
                        else:
                            file_type = "emsoft_hdf5"
                    except KeyError:
                        try:
                            manufacturer = f["manufacturer"][:][0].decode()
                            if manufacturer != "kikuchipy":
                                raise KeyError
                            else:
                                file_type = "kpy_hdf5"
                        except KeyError:
                            raise IOError("{filename} is not supported format")

    return file_type


def get_hkl_metadata(filename: str) -> dict:
    """
    Retrieve relevant EBSD metadate from Oxford/hkl exported raw patterns in tiff format, including
    pattern/projection centre position, sample pre-tilt angle, electron beam voltage, etc.

    Parameters
    ----------
    filename
        Path and file name of EBSD exported tiff image.

    Returns
    -------
    ebsd_metadata
        Metadata dictionary.

    """
    with TiffFile(filename) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
        image = tif.pages[0].asarray()

        raw_metadata = tif_tags["51122"]
        myroot = ET.fromstring(raw_metadata)

        ebsd_metadata = {
            myroot[0][i].tag: myroot[0][i].text for i in range(len(myroot[0]))
        }
        ebsd_metadata["image_type"] = myroot.tag
    return ebsd_metadata


def get_projection_centre(filename: str) -> dict:
    """
    Retrieve EBSD pattern projection centre information from relevant files

    Supported vendors are:
    * Oxford/HKL .ctf file
    * Oxford/HKL raw pattern .tiff file
    * Oxford/HKL .hoina file
    * EDAX .ang file
    * EDAX .hdf5 file

    Parameters
    ----------
    filename
        Path and file name.

    Returns
    -------
    ebsd_metadata
        Metadata dictionary.

    """
    file_type = get_file_footprint(filename)

    # * Oxford/HKL raw pattern .tiff file:        "oina-image"
    if file_type == "oina-image":
        ebsp_metadata = get_hkl_metadata(filename)
        PCx = float(ebsp_metadata["pattern-center-x-pu"])
        PCy = float(ebsp_metadata["pattern-center-y-pu"])
        PCz = float(ebsp_metadata["detector-distance-pu"])
        PC = [PCx, PCy, PCz]

    # * Oxford/HKL .hoina file:                   "hkl_hdf5"
    elif file_type == "hkl_hdf5":
        with h5.File(filename, "r") as f:
            PCx = np.mean(f["1/EBSD/Data/Pattern Center X"][:])
            PCy = np.mean(f["1/EBSD/Data/Pattern Center Y"][:])
            PCz = np.mean(f["1/EBSD/Data/Detector Distance"][:])
            PC = [PCx, PCy, PCz]

    # * EDAX .ang file                            "tsl_ang"
    elif file_type == "tsl_ang":
        with open(
            filename,
        ) as f:
            line = f.readline()
            PCx = None
            PCy = None
            PCz = None
            while line.startswith("#"):
                line = f.readline()
                if "x-star" in line:
                    PCx = float(re.split("\s+", line)[2])
                if "y-star" in line:
                    PCy = float(re.split("\s+", line)[2])
                if "z-star" in line:
                    PCz = float(re.split("\s+", line)[2])

            if all(v is not None for v in [PCx, PCy, PCz]):
                PC = [PCx, PCy, PCz]
            else:
                PC = None
                raise IOError("The file does not contain pattern centre information")

    # * EDAX .hdf5 file                           "tsl_hdf5"
    # * EMsoft .hdf5 file                         "emsoft_hdf5"
    # * Kikuchipy .hdf5 file                      "kpy_hdf5"
    elif file_type in ["tsl_hdf5", "emsoft_hdf5", "kpy_hdf5"]:
        with h5.File(filename, "r") as f:
            PCx = f["Scan 1/EBSD/Header/xpc"][:][0]
            PCy = f["Scan 1/EBSD/Header/ypc"][:][0]
            PCz = f["Scan 1/EBSD/Header/zpc"][:][0]
            PC = [PCx, PCy, PCz]

    else:
        pc = None
        raise ValueError("No pattern centre has been loaded.")

    return PC


def load_xmap(filename: str) -> crystal_map:
    """
    Load Oxford/HKL .ctf file or EDAX/TSL .ang file as Orix crystal map object

    Supported vendors are:
    * Oxford/HKL .ctf file
    * EDAX .ang file
    TODO: add support to formats Oxford ".h5oina" file, EDAX hdf5, EMsoft hdf5, Kikuchipy hdf5

    Parameters
    ----------
    filename
        Path and file name.

    Returns
    -------
        Orix crystal_map object
    """
    file_type = get_file_footprint(filename)

    if file_type == "hkl_ctf":
        xmap = ctf_reader(filename=filename)
    elif file_type == "tsl_ang":
        xmap = ang_reader(filename=filename)
    else:
        xmap = None
        raise IOError("The imported file format is not supported.")

    return xmap


# def loadctf(file_string: str) -> Rotation:
#     """Load ``.ctf`` files.

#     Parameters
#     ----------
#     file_string
#         Path to the ``.ctf`` file. This file is assumed to list the
#         Euler angles in the Bunge convention in the columns 5, 6, and 7.
#         The starting row for the data that contains Euler angles is relevant
#         to the number of inlcuded phases.

#     Returns
#     -------
#     rotation
#         Rotations in the file.
#     """
#     with open(file_string, "r") as file:
#         all_data = [line.strip() for line in file.readlines()]
#         phase_num = int(all_data[12].split("\t")[1])

#     data = np.loadtxt(file_string, skiprows=(14 + phase_num))[:, 5:8]
#     euler = np.radians(data)
#     return Rotation.from_euler(euler)


def get_avg_orientation(filename: str) -> Rotation:
    """
    Compute the average orientation of a crystal map in rotations. This is used for getting the average rotation of
    reference Si single crystal sample. The output rotation is in the TSL/EDAX convention  of crystal reference frame,
    e.g. the crystal Rotation from HKL/Oxford file is converted to the TSL convention by applying a clockwise rotation
    of 90 degree about Z axis.

    Supported vendors are:
    * Oxford/HKL .ctf file
    * EDAX .ang file

    Parameters
    ----------
    filename
        Path and file name.

    Returns
    -------
        Orix Rotation object
    """
    xmap = load_xmap(filename)
    avg_rotation = xmap.rotations.mean()

    ebsd_vendor = get_file_footprint(filename)
    if ebsd_vendor == "hkl_ctf":
        avg_rotation = avg_rotation * Rotation.from_axes_angles([0, 0, 1], -np.pi / 2)
    elif ebsd_vendor == "tsl_ang":
        avg_rotation = avg_rotation
    else:
        raise ValueError("Input file format is not supported!")

    return avg_rotation


def save_align_points(points1: np.ndarray, points2: np.ndarray) -> None:
    """
    Save the alignment points to a pkl file.

    Parameters
    ----------
    data
        A list of alignment points in the format of points1, points2
        point1 are from the reference image
        point2 are from the image to be transformed

    Returns
    -------
    None
    """
    with open("alignment_points.pkl", "wb") as data:
        pickle.dump([points1, points2], data)


def load_prev_align_points() -> np.ndarray:
    """
    Load the alignment points from a pkl file.

    Parameters
    ----------

    Returns
    -------
    [point1, point2]
        A list of alignment points in the format of [point1, point2]
        point1 are from the reference image
        point2 are from the image to be transformed

    """
    with open("alignment_points.pkl", "rb") as data:
        [points1, points2] = pickle.load(data)
    points2 = np.array(points2, dtype=np.float32)
    points1 = np.array(points1, dtype=np.float32)

    return [points1, points2]
