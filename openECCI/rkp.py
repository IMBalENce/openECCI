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

from openECCI.io import get_sem_metadata
import numpy as np
from diffsims.crystallography import ReciprocalLatticeVector
from orix import plot, crystal_map
from orix.crystal_map import Phase
from orix.quaternion import Rotation
from orix.vector import Vector3d
import matplotlib.pyplot as plt
import math
import kikuchipy as kp


def get_sim_rkp(
    RKP_masterpattern,
    xtal_rotation: Rotation,
    st_rot_angle: float,
    st_tilt_angle: float,
    corr_angles: list,
    ref_ECP: str,
    cam_length: float = 4,
    RKP_shape: list = None,
):
    """
    With a given crystal orientation (in Orix Rotation format), SEM stage rotation, tilt angle,
    and RKP correction angles, this function computes and returns RKP (Reflected Kikuchi pattern)
    using EBSD master pattern generated from EMsoft software package. The default virtual RKP
    detector is defined as a detector located below SEM pole piece and perpendicular to the electon
    beam with a pixel dimension of 10 micron and binning 1.

    Parameters
    ----------
    RKP_masterpattern
        Numpy 2d array of image 1.
    xtal_rotation
        Numpy 2d array of image 2.
    st_rot_angle
        SEM stage rotation angle in degrees following right hand rule.
    st_tilt_angle
        SEM stage tilt angle in degrees.
    corr_angles
        A list of three correction coefficients [x_corr, y_corr, z_corr] to align RKP simulation
        with experimental ECP.
    ref_ECP
        Reference ECP pattern acquired from SEM for calibration. Image resolution without databar
        is used for generating RKP with an identical detector shape.
    cam_length
        cam_length is actually the pattern center PC\ :sub:`z` defined as the distance from RKP
        virtual detector scintillator to the sample divided by the pattern height physical dimension.
    RKP_shape
        Shape of RKP pixel dimension in the form of list [x, y]

    Returns
    -------
    rkp_sim
        A kikuchipy EBSD object that contains simulated RKP pattern
    """

    ecp_metadata = get_sem_metadata(filename=ref_ECP)
    if RKP_shape is None:
        rkp_shape = ecp_metadata["resolution"]
    else:
        rkp_shape = RKP_shape
    HV = ecp_metadata["HV"]

    st_rot = Rotation.from_axes_angles([0, 0, 1], -st_rot_angle, degrees=True)
    st_tilt = Rotation.from_axes_angles([0, 1, 0], -st_tilt_angle, degrees=True)

    tiltX_corr = Rotation.from_axes_angles([1, 0, 0], -corr_angles[0], degrees=True)
    tiltY_corr = Rotation.from_axes_angles([0, 1, 0], -corr_angles[1], degrees=True)
    tiltZ_corr = Rotation.from_axes_angles([0, 0, 1], -corr_angles[2], degrees=True)

    rkp_detector = kp.detectors.EBSDDetector(
        shape=(rkp_shape[1], rkp_shape[0]),
        tilt=-90,
        sample_tilt=0,
        pc=[0.5, 0.5, cam_length],
        # pc=[0.5, 0.5, 8.5], # for camera length for Zeiss
        px_size=10,
        binning=1,
    )

    rkp_sim = RKP_masterpattern.get_patterns(
        rotations=xtal_rotation
        * tiltZ_corr
        * tiltY_corr
        * tiltX_corr
        * st_rot
        * st_tilt,
        detector=rkp_detector,
        energy=HV,
        compute=True,
        show_progressbar=False,
    )

    return rkp_sim


def interactive_xmap(
    xmap: crystal_map, phase_name: str, overlay: str = None, hybrid_mode: bool = False
):
    """
    Interactive_xmap
    requires pyqt to display the ipf map.

    Parameters
    ----------
    xmap
        Numpy 2d array of image 1.
    phase_name
        The name of phase to display in the IPF map. This could be listed by printing the xmap
        object and return all included phase names
    overlay
        This could be the additional properties included in xmap object (after printing the xmap
        object, the available items are listed in the Properties). By including an overlay could
        enrich the information plotted in the IPF map. For example, adding overy="BC" for an
        Oxford/HKL IPF map will overlay the band contrast information to the map and enhance the
        contrast from grain boundaries.
    hybrid_mode
        if hybrid_mode is enabled, EBSD ipf map will be displayed as an overlay on top of a SEM
        image

    Returns
    -------
    coords
        A list of all clicked coordinates. The last two elements are the [x,y] coordinate for the
        last clicked point.

    TODO: for hybrid mode, the EBSD IPF map will be aligned with the SEM BSE iamge and transformed
    as an overlay on top of the SEM image. The click position on the transformed image need to be
    converted back (inverse of the homography transformation matrix) to the pixel position in the
    original EBSD IPF map to retrieve the Euler angles.

    """

    xmap_image = get_xmap_image(xmap, phase_name, overlay)

    fig, ax = plt.subplots(subplot_kw=dict(projection="plot_map"), figsize=(12, 8))
    ax.imshow(xmap_image)
    ax.set_title("Double click to get Euler angles")

    coords = []

    def on_click(event):
        if event.dblclick:
            # print(event.xdata, event.ydata)
            coords.append(event.xdata)
            coords.append(event.ydata)

            plt.clf()
            plt.imshow(xmap_image)
            try:
                x_pos = coords[-2]
                y_pos = coords[-1]
            except:
                x_pos = 0
                y_pos = 0
            [Eu1, Eu2, Eu3] = np.rad2deg(
                Rotation.to_euler(xmap[int(y_pos), int(x_pos)].orientations)
            )[0]
            plt.plot(x_pos, y_pos, "+", c="black", markersize=12, markeredgewidth=2)
            plt.title(
                f"Euler angles: {np.round(Eu1, 2)}, {np.round(Eu2, 2)}, {np.round(Eu3, 2)}"
            )

            plt.draw()
            plt.axis("off")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    plt.draw()
    plt.axis("off")
    plt.tight_layout()

    return coords  # coordintes in x, y format


def rkp_angle_finder(
    RKP_masterpattern,
    xtal_rotation: Rotation,
    phase_name: str,
    ref_ECP: str,
    corr_angles: list,
    st_rot_angle: float = 0,
    st_tilt_angle: float = 0,
    cam_length: float = 4,
    RKP_shape: list = None,
    stage_mode: str = "rot-tilt",
):
    """
    Return the required SEM stage rotation and tilt so as to align the clicked direction in angular space with electron beam
    direction.

    Parameters
    ----------
    RKP_masterpattern

    xtal_rotation

    ref_ECP

    corr_angles

    st_rot_angle

    st_tilt_angle

    stage_mode

    Returns
    -------
    coords
        A list of all clicked coordinates. The last two elements are the [x,y] coordinate for the
        last clicked point.

    """
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, subplot_kw=dict(projection="plot_map"), figsize=(16, 8)
    )
    st_rot = Rotation.from_axes_angles([0, 0, 1], -st_rot_angle, degrees=True)
    st_tilt = Rotation.from_axes_angles([0, 1, 0], -st_tilt_angle, degrees=True)

    # Plot the large angular range RKP to give an overview and kikuchi band/pole labelling
    ax1.cla()
    ref = ReciprocalLatticeVector(
        phase=phase_name, hkl=[[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]]
    )
    ref = ref.symmetrise().unique()
    hkl_sets = ref.get_hkl_sets()
    hkl_sets
    simulator = kp.simulations.KikuchiPatternSimulator(ref)

    sim_RKP_lowMag = get_sim_rkp(
        RKP_masterpattern,
        xtal_rotation,
        st_rot_angle,
        st_tilt_angle,
        corr_angles,
        ref_ECP,
        cam_length=0.8,
        RKP_shape=RKP_shape,
    )
    sim_RKP_lowMag_pattern = np.squeeze(sim_RKP_lowMag.data)
    sim = simulator.on_detector(sim_RKP_lowMag.detector, sim_RKP_lowMag.xmap.rotations)
    ax1.imshow(sim_RKP_lowMag_pattern, cmap="gray")
    ax1.set_title("RKP Overview with indexing", loc="center")
    ax1.axis("off")

    lines, zone_axes, zone_axes_labels = sim.as_collections(
        zone_axes=True,
        zone_axes_labels=True,
        zone_axes_labels_kwargs=dict(fontsize=12),
    )
    ax1.add_collection(lines)
    ax1.add_collection(zone_axes)
    for label in zone_axes_labels:
        ax1.add_artist(label)

    # Plot a second RKP with smaller angular range to show more detailed kikuchi band close to projection centre
    ax2.cla()
    sim_RKP_hiMag = get_sim_rkp(
        RKP_masterpattern,
        xtal_rotation,
        st_rot_angle,
        st_tilt_angle,
        corr_angles,
        ref_ECP,
        cam_length=cam_length,
        RKP_shape=RKP_shape,
    )
    sim_RKP_hiMag_pattern = np.squeeze(sim_RKP_hiMag.data)
    ecp_dim = list(sim_RKP_hiMag_pattern.shape)
    # get the BKP detector physcial dimensions
    [PCx_rkp, PCy_rkp, PCz_rkp] = sim_RKP_hiMag.detector.pc[0]
    [Ny, Nx] = sim_RKP_hiMag.detector.shape
    px_size_rkp = sim_RKP_hiMag.detector.px_size
    binning_rkp = sim_RKP_hiMag.detector.binning

    ax2.imshow(sim_RKP_hiMag_pattern, cmap="gray")
    # plot the centre marker for rkp prjection centre
    ax2.plot(PCx_rkp * Nx, PCy_rkp * Ny, "+", c="red", markersize=15, markeredgewidth=3)

    ax2.axis("off")
    ax2.set_title("Click on RKP to find required stage movements", loc="center")
    # ax2.set_title(f"Eular angle ({fe_euler_rotation.to_euler(degrees=True)}), required st rot {st_rot_target}, required st tilt {st_tilt_target}\n", fontsize=10)
    coords = []

    def on_click2(event):
        if event.dblclick:
            # print(event.xdata, event.ydata)
            coords.append(event.xdata)
            coords.append(event.ydata)

            # plt.clf()
            ax2.cla()

            ax2.imshow(sim_RKP_hiMag_pattern, cmap="gray")
            try:
                x_pos = coords[-2]
                y_pos = coords[-1]
            except:
                x_pos = 0
                y_pos = 0
            ax2.plot(x_pos, y_pos, "+", c="yellow", markersize=15, markeredgewidth=3)
            ax2.plot(
                PCx_rkp * Nx,
                PCy_rkp * Ny,
                "+",
                c="red",
                markersize=15,
                markeredgewidth=3,
            )

            # Calculated the phycial distances on the detector from selected pixel to the projection centre,
            # calculated in um (micrometer)
            [coord_x, coord_y] = [x_pos, y_pos]
            distance_x = (coord_x - PCx_rkp * Nx) * px_size_rkp * binning_rkp
            distance_y = (PCy_rkp * Ny - coord_y) * px_size_rkp * binning_rkp
            distance_l = PCz_rkp * Ny * px_size_rkp * binning_rkp

            if stage_mode == "rot-tilt":
                # if the native SEM rotation tilt stage is used, calcuated the azimuthal and polar angle
                # of the selected pixel in the detector frame
                azi_rkp = np.arctan2(distance_x, distance_y)
                polar_rkp = math.atan(
                    math.sqrt(distance_x**2 + distance_y**2) / distance_l
                )
                ax2.set_title(
                    f"""
                    Pixel position {int(x_pos)}, {int(y_pos)}
                    Physical distance on RKP detector {round(distance_x,2), round(distance_y,2), round(distance_l,2)}um
                    Stage Rot {round(math.degrees(azi_rkp),2)}\N{DEGREE SIGN}, Stage tilt {round(math.degrees(polar_rkp),2)}\N{DEGREE SIGN}""",
                    fontsize=11,
                    loc="left",
                )
            elif stage_mode == "double-tilt":
                #
                theta_x_rkp = math.atan(distance_x / distance_l)
                theta_y_rkp = math.atan(distance_y / distance_l)
                ax2.set_title(
                    f"""
                        Pixel position {int(x_pos)}, {int(y_pos)}
                        Physical distance on RKP detector {round(distance_x,2), round(distance_y,2), round(distance_l,2)}um
                        Tilt around Xs {round(math.degrees(theta_x_rkp),2)}\N{DEGREE SIGN}, Tilt around Ys {round(math.degrees(theta_y_rkp),2)}\N{DEGREE SIGN}""",
                    fontsize=11,
                    loc="left",
                )
            plt.draw()
            ax2.axis("off")

    fig.canvas.mpl_connect("button_press_event", on_click2)
    plt.show()
    plt.draw()
    plt.axis("off")

    return coords  # coordintes in x, y format


def get_xmap_image(
    xmap: crystal_map,
    phase_name: str,
    overlay: str = None,
) -> np.ndarray:
    """
    Get the EBSD IPF map image in numpy array with or without overlay for plotting purposes.

    Parameters
    ----------
    xmap
        Numpy 2d array of image 1.
    phase_name
        The name of phase to display in the IPF map. This could be listed by printing the xmap
        object and return all included phase names
    overlay
        This could be the additional properties included in xmap object (after printing the xmap
        object, the available items are listed in the Properties). By including an overlay could
        enrich the information plotted in the IPF map. For example, adding overy="BC" for an
        Oxford/HKL IPF map will overlay the band contrast information to the map and enhance the
        contrast from grain boundaries.

    Returns
    -------
    xmap_image
        A numpy 3d array of the EBSD IPF color map

    """

    ckey = plot.IPFColorKeyTSL(
        xmap.phases[phase_name].point_group, direction=Vector3d.zvector()
    )
    rgb_phase = ckey.orientation2color(xmap[phase_name].orientations)
    rgb_all = np.zeros((xmap.size, 3))
    rgb_all[xmap.phase_id == xmap.phases.id_from_name(phase_name)] = rgb_phase

    if overlay is not None:
        xmap_overlay = rgb_all.reshape(xmap.shape + (3,))
        overlay_1dim = (xmap.prop[overlay]).reshape(xmap.shape)
        overlay_min = np.nanmin(overlay_1dim)
        rescaled_overlay = (overlay_1dim - overlay_min) / (
            np.nanmax(overlay_1dim) - overlay_min
        )
        n_channels = 3
        for i in range(n_channels):
            xmap_overlay[:, :, i] *= rescaled_overlay
        xmap_image = xmap_overlay
    else:
        xmap_image = rgb_all.reshape(xmap.shape + (3,))

    return xmap_image
