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

from src.util import normalize, rotate_image_around_point
from src import io, rkp
import numpy as np
from orix.quaternion import Rotation
from diffsims.crystallography import ReciprocalLatticeVector
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
import kikuchipy as kp
import math
import matplotlib.patches as patches


def modal_assurance_criterion(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute the MAC (modal assurance criterion) coefficient between image1 and image2. MAC coefficient is
    a statistical indicator that is most sensitive to large differences and relatively insensitive to small
    differences in the mode shapes, e.g. image1 and image2. The returned value should be in the range [0, 1].
    Greater MAC means higher similarity.

    Computers and Structures 74 (2000) 375-383
    The Modal Assurance Criterion-Twenty Years of Use and Abuse, Sound and Vibrations 2003

    Parameters
    ----------
    image1
        Numpy 2d array of image 1
    image2
        Numpy 2d array of image 2

    Returns
    -------
    MAC
        A float in the range [0, 1]
    """
    image1 = normalize(image1)
    image2 = normalize(image2)
    m, n = image1.shape
    I1 = 0
    I2 = 0
    I3 = 0
    for ii in range(m):
        I1 = I1 + np.dot(image1[ii, :], np.conj(image2[ii, :]))
        I2 = I2 + np.dot(image1[ii, :], np.conj(image1[ii, :]))
        I3 = I3 + np.dot(image2[ii, :], np.conj(image2[ii, :]))

    MAC = np.abs(I1) ** 2 / (I2 * I3)
    return MAC


class orientation_calibration:
    def __init__(
        self,
        corr_angles: list,
        PCz: float,
        reference_ECP_path: str,
        si_master_pattern,
        Si_xtal: Rotation,
    ):
        """
        A procedure to calibrate and minimise the deviation between experimental Electron Channelling Pattern (ECP)
        from a calibration single crystal material (Si[001] wafer is typically used) and simulated Reflected Kikuchi
        Pattern (RKP) from its EBSD mapping Euler angles. During the process, the three "active" rotation correction
        angles and the camera length, PCz, are optimized to minimize the deviation.

        params
        corr_angles
            A list of rotation correction angles in degrees [tiltX_corr_angle, tiltY_corr_angle, tiltZ_corr_angle].
        PCz
            One of the pattern/projection centre values for the virtual RKP detector that is relavent to camera length
            (sample/detector distance).
        reference_ECP_path
            The reference image path of the ECP pattern acquired from Si[001] wafer at the beginning of ECCI experiment.
        si_master_pattern
            The master pattern of Si generated using the corresponding electron energy from EMsoft.
        Si_xtal
            Orientation of the calibration Si single crystal acquired from EBSD and in the form of Orix.quaternion.rotation.
        """

        self.corr_angles = corr_angles
        self.reference_ECP_path = reference_ECP_path
        self.master_pattern = si_master_pattern
        self.Si_xtal = Si_xtal
        self.cam_length = PCz
        self.initial_guess = {
            "tiltX_corr_angle": self.corr_angles[0],
            "tiltY_corr_angle": self.corr_angles[1],
            "tiltZ_corr_angle": self.corr_angles[2],
            "PCz": self.cam_length,
        }

        print(
            f"Orientation Calibration Object created using the following parameters: \
              \nInitial guess corrections: {self.initial_guess},\
              \nReference ECP: {self.reference_ECP_path},\
              \nMaster Pattern: {self.master_pattern}"
        )

    def __repr__(self):
        return f"Orientation Calibration Object: {self.corr_angles}, {self.reference_ECP_path}, {self.master_pattern}"

    def _get_minus_MAC(self, parameter_list: list) -> float:
        """
        Compute the 1-MAC (Modal Assurance Criterion) coefficient between the experiment ECP and simulated RKP.

        Parameters
        ----------
        parameter_list
            A list of parameters requires optimization [tiltX_corr_angle, tiltY_corr_angle, tiltZ_corr_angle, PCz]
            Note, the value of MAC, in the range [0, 1], represents how similar two images are, i.e. the closer
            the value to 1, the closer match the two images are. By using 1-MAC, it reverses the relationship.

        Returns
        -------
        1-MAC
            A float in the range [0, 1]
        """

        ecp_metadata = io.get_sem_metadata(self.reference_ECP_path)
        ecp_shape = ecp_metadata["resolution"]
        if ecp_metadata["databar"] == True:
            exp_ECP = plt.imread(self.reference_ECP_path)[: ecp_shape[1], :]
        else:
            exp_ECP = plt.imread(self.reference_ECP_path)

        si_average_euler_rotation = self.Si_xtal

        sim_rkp = rkp.get_sim_rkp(
            RKP_masterpattern=self.master_pattern,
            xtal_rotation=si_average_euler_rotation,
            st_rot_angle=0,
            st_tilt_angle=0,
            corr_angles=parameter_list[0:3],
            ref_ECP=self.reference_ECP_path,
            cam_length=parameter_list[3],
            RKP_shape=ecp_shape,
        )

        MAC = 1 - modal_assurance_criterion(exp_ECP, np.squeeze(sim_rkp.data))
        return MAC

    def _get_minus_NDP(self, parameter_list: list) -> float:
        """
        Compute the 1-NDP (Normalized Dot Product) coefficient between the experiment ECP and simulated RKP.
        Note, the value of NDP, in the range [0, 1], represents how similar two images are, i.e. the closer
        the value to 1, the closer match the two images are. By using 1-NDP, it reverses the relationship.

        Parameters
        ----------
        parameter_list
            A list of parameters requires optimization [tiltX_corr_angle, tiltY_corr_angle, tiltZ_corr_angle, PCz]

        Returns
        -------
        1-NDP
            A float in the range [0, 1]
        """
        ecp_metadata = io.get_sem_metadata(self.reference_ECP_path)
        ecp_shape = ecp_metadata["resolution"]
        if ecp_metadata["databar"] == True:
            exp_ECP = plt.imread(self.reference_ECP_path)[: ecp_shape[1], :]
        else:
            exp_ECP = plt.imread(self.reference_ECP_path)

        si_average_euler_rotation = self.Si_xtal

        sim_rkp = rkp.get_sim_rkp(
            RKP_masterpattern=self.master_pattern,
            xtal_rotation=si_average_euler_rotation,
            st_rot_angle=0,
            st_tilt_angle=0,
            corr_angles=parameter_list[0:3],
            ref_ECP=self.reference_ECP_path,
            cam_length=parameter_list[3],
            RKP_shape=ecp_shape,
        )

        NDP = 1 - np.dot(
            normalize(exp_ECP.flatten()), normalize(np.squeeze(sim_rkp.data).flatten())
        )

        return NDP

    def optimize_calibration(self, method="NDP"):
        """
        By using the Nelder-Mead method, the optimized parameters, i.e. [tiltX_corr_angle, tiltY_corr_angle, tiltZ_corr_angle, PCz],
        can be retrieved by minimizing the deviation between experimental ECP and simulated RKP (minimize 1-NDP or 1-MAC).

        Parameters
        ----------
        method
            NDP (Normalized Dot Product) or MAC (Modal Assurance Criterion) methods can be used. By default, NDP is selected.

        """
        # tiltX_corr_angle, tiltY_corr_angle, tiltZ_corr_angle, PCz

        initial_guess = [
            self.initial_guess["tiltX_corr_angle"],
            self.initial_guess["tiltY_corr_angle"],
            self.initial_guess["tiltZ_corr_angle"],
            self.initial_guess["PCz"],
        ]
        if method == "MAC":
            res = minimize(
                self._get_minus_MAC,
                initial_guess,
                method="Nelder-Mead",
                tol=1e-6,
                options={"disp": True},
            )
            result = res.x
            print("Rotation calibration optimized using MAC method")
            print(
                f"tiltX_corr_angle: {result[0]}, tiltY_corr_angle: {result[1]}, tiltZ_corr_angle: {result[2]} PCz: {result[3]}"
            )
            print(f"MAC: {1-self._get_minus_MAC(result)}")
            return {
                "tiltX_corr_angle": result[0],
                "tiltY_corr_angle": result[1],
                "tiltZ_corr_angle": result[2],
                "PCz": result[3],
            }

        elif method == "NDP":
            res = minimize(
                self._get_minus_NDP,
                initial_guess,
                method="Nelder-Mead",
                tol=1e-6,
                options={"disp": True},
            )
            result = res.x
            print("Rotation calibration optimized using NDP method")
            print(
                f"tiltX_corr_angle: {result[0]}, tiltY_corr_angle: {result[1]}, tiltZ_corr_angle: {result[2]} PCz: {result[3]}"
            )
            print(f"NDP: {1-self._get_minus_NDP(result)}")
            return {
                "tiltX_corr_angle": result[0],
                "tiltY_corr_angle": result[1],
                "tiltZ_corr_angle": result[2],
                "PCz": result[3],
            }

        else:
            raise ValueError(
                "Current supported methods are 'MAC' and 'NDP', please choose the supported method."
            )


class convergence_angle_measurement:
    def __init__(
        self,
        aperture_image_path: str,
    ):
        """
        A process to measure the electron beam convergence (semi) angle
        """
        self.img_metadata = io.get_sem_metadata(aperture_image_path)
        img_resolution = self.img_metadata["resolution"]

        img = cv2.imread(aperture_image_path, cv2.IMREAD_ANYDEPTH)
        img_gray = img[: img_resolution[1], : img_resolution[0]]
        self.image = img_gray
        self.img_resolution = img_resolution
        self.data_dir, self.filename = os.path.split(aperture_image_path)

    def __repr__(self):
        return f"Convergence Angle Measurement Object: {self.Pt_aperture_image_path}"

    def find_centroid(self, threshold: float, plot: bool = True):
        """
        find the centroid of the aperture image without any rotation.
        """
        ret, thresh = cv2.threshold(
            self.image, threshold, np.max(self.image), cv2.THRESH_BINARY_INV
        )

        M = cv2.moments(thresh, False)
        # calculate x,y coordinate of circle center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if plot:
            plt.figure()
            plt.imshow(self.image, cmap="gray")
            plt.imshow(thresh, alpha=0.3)
            plt.scatter(cX, cY, c="r")
            plt.show()

        self.original_centroid = [cX, cY]
        # reduce the profile range by 10 pixels to avoid the edge of the image
        self.profile_range = np.min(self._get_distance_to_edge([cX, cY])) - 10

        return [cX, cY]

    def _get_edge_profile(
        self,
        aperture_image: np.ndarray,
        centroid: list,
        filter_sigma: float = 5,
        plot: bool = False,
    ):
        """
        Get the edge profile from  aperture image along the horizontal line from the centroid to the right
        """
        profile = aperture_image[
            int(centroid[1]), int(centroid[0]) : (int(centroid[0]) + self.profile_range)
        ]
        normalized_profile = normalize(profile)
        gradient_1st = gaussian_filter(
            np.gradient(normalized_profile), sigma=filter_sigma
        )
        gradient_2nd = gaussian_filter(np.gradient(gradient_1st), sigma=filter_sigma)
        normalized_gradient_2nd = normalize(gradient_2nd)

        x = (
            np.arange(0, len(normalized_profile))
            * self.img_metadata["pixel_size"]
            * 1e6
        )

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(x, normalized_profile, label="Profile", color="royalblue")
            plt.plot(x, normalize(gradient_1st), label="1st derivative", color="orange")
            plt.plot(x, normalize(gradient_2nd), label="2nd derivative", color="green")
            plt.xlabel("Distance (um)")
            plt.ylabel("Intensity (a.u.)")
            plt.legend()

            plt.scatter(
                x[np.argmax(gradient_2nd)], np.max(normalized_gradient_2nd), color="b"
            )
            plt.scatter(
                x[np.argmin(gradient_2nd)], np.min(normalized_gradient_2nd), color="b"
            )
            plt.scatter(
                x[np.argmax(gradient_2nd)],
                normalized_profile[np.argmax(gradient_2nd)],
                color="r",
            )
            plt.scatter(
                x[np.argmin(gradient_2nd)],
                normalized_profile[np.argmin(gradient_2nd)],
                color="r",
            )

            plt.show()

        return normalized_profile, gradient_1st, gradient_2nd

    def _get_distance_to_edge(self, point_coord: list):
        """
        Get the distance from the centroid to the edge of the initial unrotated image in the four directions.
        """
        [cX, cY] = point_coord
        return [cX, cY, self.img_resolution[0] - cX, self.img_resolution[1] - cY]

    def get_profile_from_angle(
        self,
        angle: float,
        filter_sigma: float = 5,
        plot: bool = False,
    ):
        """
        Get the edge profile from the aperture image from the centroid of the aperturn towards a given angle.
        The zero degree angle is defined as the horizontal line from the centroid to the right. The positive angle
        is in the counter-clockwise direction.
        """
        centroid = self.original_centroid
        rotated_img, new_centroid = rotate_image_around_point(
            self.image, centroid, -angle
        )
        profile, gradient_1st, gradient_2nd = self._get_edge_profile(
            rotated_img, new_centroid, filter_sigma, plot=plot
        )

        return profile, gradient_1st, gradient_2nd

    def _conv_angle_from_profile(
        self, gradient_2nd: np.ndarray, height_offset: float = None
    ):
        """
        Compute the convergence angle from the 2nd derivative of the edge profile.
        """
        if height_offset is None:
            height_offset = self.img_metadata["stage_z"] - self.img_metadata["WD"]

        probe_diameter = (
            np.argmin(gradient_2nd) - np.argmax(gradient_2nd)
        ) * self.img_metadata[
            "pixel_size"
        ]  # in m
        tan_alpha = (probe_diameter / 2) / height_offset
        conv_half_angle = np.arctan(tan_alpha) * 1000  # in mrad

        return conv_half_angle, probe_diameter

    def compute(
        self,
        angle_step: float,
        filter_sigma: float = 5,
        plot: bool = True,
        save_results: bool = False,
    ):
        """
        Compute the gradient of the edge profile of the aperture image along different angles.
        """
        steps = np.arange(0, 359, angle_step)

        for index, value in enumerate(tqdm(steps)):
            profile, gradient_1st, gradient_2nd = self.get_profile_from_angle(
                angle=value, filter_sigma=filter_sigma, plot=False
            )
            conv_half_angle, probe_diameter = self._conv_angle_from_profile(
                gradient_2nd
            )

            if index == 0:
                profiles = profile
                gradients_1st = gradient_1st
                gradients_2nd = gradient_2nd
                conv_half_angles = conv_half_angle
                probe_diameters = probe_diameter
            else:
                profiles = np.vstack((profiles, profile))
                gradients_1st = np.vstack((gradients_1st, gradient_1st))
                gradients_2nd = np.vstack((gradients_2nd, gradient_2nd))
                conv_half_angles = np.append(conv_half_angles, conv_half_angle)
                probe_diameters = np.append(probe_diameters, probe_diameter)

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=[14, 10])

            x = np.arange(0, profiles.shape[1]) * self.img_metadata["pixel_size"] * 1e6
            profile_plot = profiles[0, :]
            normalized_profile_plot = normalize(profile_plot)
            normalized_gradient_2nd_plot = normalize(gradients_2nd[0, :])

            [cX, cY] = self.original_centroid
            axes[0, 0].imshow(self.image, cmap="gray")
            axes[0, 0].scatter(cX, cY, c="r")
            axes[0, 0].set_title("Defocused Aperture Image")

            axes[0, 1].plot(
                x, normalized_profile_plot, label="Profile", color="royalblue"
            )
            axes[0, 1].plot(
                x,
                normalize(gradients_1st[0, :]),
                "-",
                label="1st derivative",
                color="orange",
            )
            axes[0, 1].plot(
                x,
                normalize(gradients_2nd[0, :]),
                "-",
                label="2nd derivative",
                color="green",
            )
            axes[0, 1].scatter(
                x[np.argmax(normalized_gradient_2nd_plot)],
                np.max(normalized_gradient_2nd_plot),
                color="b",
            )
            axes[0, 1].scatter(
                x[np.argmin(normalized_gradient_2nd_plot)],
                np.min(normalized_gradient_2nd_plot),
                color="b",
            )
            axes[0, 1].scatter(
                x[np.argmax(normalized_gradient_2nd_plot)],
                profiles[0, :][np.argmax(normalized_gradient_2nd_plot)],
                color="r",
            )
            axes[0, 1].scatter(
                x[np.argmin(normalized_gradient_2nd_plot)],
                profiles[0, :][np.argmin(normalized_gradient_2nd_plot)],
                color="r",
            )
            axes[0, 1].set_xlabel("Distance (um)")
            axes[0, 1].set_ylabel("Intensity (a.u.)")
            axes[0, 1].legend()

            axes[1, 0].plot(
                steps,
                probe_diameters * 1e6,
                "x",
                color="b",
                label="Diameter of diverged probe disk",
            )
            axes[1, 0].set_ylim([5, 60])
            # axes[1,0].set_title(f'Diameter of diverged probe disk vs. angle')
            axes[1, 0].set_xlabel("Angle of measurement (deg)")
            axes[1, 0].set_ylabel("Diverged probe disk diameter (um)")
            ax2 = axes[1, 0].twinx()
            ax2.plot(steps, conv_half_angles, "o", color="r", label="Convergence angle")
            ax2.set_ylabel("Convergence angle (mrad)")
            ax2.set_ylim([0, 15])

            # Add legend
            points1, label1 = axes[1, 0].get_legend_handles_labels()
            points2, label2 = ax2.get_legend_handles_labels()
            ax2.legend(points1 + points2, label1 + label2)

            text = f"""
            Filename: {self.filename}
            Microscope vendor: {self.img_metadata['vendor']}
            High voltage: {self.img_metadata['HV']} kV
            Beam current: {round(self.img_metadata['beam_current']*1e9, 3)} nA
            Obj aperture: {round(self.img_metadata['aperture_diameter']*1e6, 3)} um
            Column mode: {self.img_metadata['column_mode']}
            StageZ: {self.img_metadata['stage_z']*1e3} mm
            WD: {self.img_metadata['WD']*1e3} mm
            Average probe diameter: {round(np.mean(probe_diameters)*1e6, 3)} um
            Average convergence half angle: {round(np.mean(conv_half_angles), 3) } mrad
            Stdev convergence half angle: {round(np.std(conv_half_angles), 3) } mrad
            Beam convergence range : {round(np.degrees(np.mean(conv_half_angles) * 2 /1000), 3)} deg
            """
            axes[1, 1].axis("off")
            axes[1, 1].text(0, 0, text, fontsize=12)

            if save_results:
                output_filename = f"HV{self.img_metadata['HV']}_curr{self.img_metadata['beam_current']*1e9}nA_{self.img_metadata['column_mode']}_Z{self.img_metadata['stage_z']*1e3}_WD{self.img_metadata['WD']*1e3}.png"
                plt.savefig(os.path.join(self.data_dir, output_filename), dpi=300)

        self.angles = steps
        self.profiles = profiles
        self.gradients_1st = gradients_1st
        self.gradients_2nd = gradients_2nd
        self.conv_half_angles = conv_half_angles
        self.probe_diameters = probe_diameters


class ipf_image_correlation:
    """
    Compute the image correlation between the reference and the warp image.
    """

    def __init__(
        self,
        reference_image_path: str,
        ebsd_file_path: str,
    ):
        """
        Compute the image correlation between the reference SEM image and the EBSD ipf map.

        Parameters
        ----------


        """
        xmap = io.load_xmap(ebsd_file_path)

        sem = cv2.imread(reference_image_path).astype("float32")
        sem_norm = cv2.cvtColor(sem, cv2.COLOR_BGR2RGB)
        sem_norm /= 255
        # height, width, _ = sem_norm.shape
        sem_shape = sem_norm.shape

        self.xmap = xmap
        self.sem_shape = sem_shape
        self.sem_norm = sem_norm
        self.reference_image_path = reference_image_path
        print(xmap)

    def get_ipf_map(self, phase_name: str, plot: bool = True):
        """
        Get the IPF map of a phase from the EBSD data.
        """
        ipf_map = rkp.get_xmap_image(
            self.xmap, phase_name=phase_name, overlay="BC"
        ).astype("float32")
        ipf_image = cv2.cvtColor(ipf_map, cv2.COLOR_BGR2RGB)

        self.ipf_map = ipf_map

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(ipf_image)
            plt.show()

        return ipf_map

    def _homography_transform(
        self,
        alignment_points_ipf: np.ndarray,
        alignment_points_ref: np.ndarray,
    ):
        """
        Compute the homography transformation matrix from the alignment points.

        Parameters
        ----------
        alignment_points_ref

        alignment_points_warp

        Returns
        -------

        """
        # Convert the list of points to numpy array

        pts_sem = np.array(alignment_points_ref, dtype=np.float32)
        pts_ipf = np.array(alignment_points_ipf, dtype=np.float32)

        homography, mask = cv2.findHomography(pts_ipf, pts_sem, 0)
        warp_ipf = cv2.warpPerspective(
            self.ipf_map, homography, (self.sem_shape[1], self.sem_shape[0])
        )
        inverse_transformation_matrix = np.linalg.inv(homography)
        return warp_ipf, inverse_transformation_matrix

    def get_ipf_points(self):
        # display the interactive EBSD IPF map to select correlation points
        # Create a function to handle mouse click events
        def onclick(event):
            if event.dblclick:
                x, y = int(event.xdata), int(
                    event.ydata
                )  # Get the pixel coordinates from the click event
                points.append((x, y))  # Store the selected point
                labels.append(len(points))  # Prompt for a label

                # Update the plot to show the selected point and label
                ax.plot(x, y, "ro", markersize=5)
                ax.text(x, y, labels[-1], fontsize=12, color="red")
                plt.draw()

        # Initialize lists to store selected points and their labels
        points = []
        labels = []

        # Create a figure and display the image
        fig, ax = plt.subplots()
        ax.imshow(self.ipf_map)
        ax.set_title("Click on the image to select points.")

        # Connect the mouse click event to the function
        cid = fig.canvas.mpl_connect("button_press_event", onclick)

        # Display the image and allow interactive selection
        plt.show()

        self.ipf_points = points

    def get_sem_points(self):
        """
        Get the clicked points from sem reference image for the correlation.
        """

        # display the interactive SEM image to select correlation points
        # Create a function to handle mouse click events
        def onclick2(event):
            if event.dblclick:
                x2, y2 = int(event.xdata), int(
                    event.ydata
                )  # Get the pixel coordinates from the click event
                points2.append((x2, y2))  # Store the selected point
                labels2.append(len(points2))  # Prompt for a label

                # Update the plot to show the selected point and label
                ax2.plot(x2, y2, "ro", markersize=5)
                ax2.text(x2, y2, labels2[-1], fontsize=12, color="red")
                plt.draw()
                plt.pause(0.0001)

        # Initialize lists to store selected points and their labels
        points2 = []
        labels2 = []

        # Create a figure and display the image
        fig2, ax2 = plt.subplots()
        ax2.imshow(self.sem_norm, cmap="gray")
        ax2.set_title("Click on the image to select points.")

        # Connect the mouse click event to the function
        cid = fig2.canvas.mpl_connect("button_press_event", onclick2)

        # Display the image and allow interactive selection
        plt.show()

        self.sem_points = points2

    def process(self, plot: bool = True):
        """
        Compute the homography transformation matrix from the selected points and apply the transformation to the IPF map.
        """
        warp_ipf, inverse_transformation_matrix = self._homography_transform(
            self.ipf_points, self.sem_points
        )

        self.warp_ipf = warp_ipf
        alpha = 0.6
        beta = 1.0 - alpha
        ipf_warp_blended = cv2.addWeighted(self.sem_norm, alpha, warp_ipf, beta, 0.0)

        if plot:
            fig = plt.subplots(figsize=(10, 8))
            plt.imshow(ipf_warp_blended)
            plt.axis("off")
            plt.tight_layout()

        self.ipf_warp_blended = ipf_warp_blended
        self.inverse_transformation_matrix = inverse_transformation_matrix

        return ipf_warp_blended, inverse_transformation_matrix

    def save_alignment_points(
        self,
        save_path: str,
    ):
        """
        Save the alignment points to a pickle file.
        """
        ipf_points = np.array(self.ipf_points, dtype=np.float32)
        sem_points = np.array(self.sem_points, dtype=np.float32)
        io.save_align_points(ipf_points, sem_points)

    def load_alignment_points(
        self,
        load_path: str,
    ):
        """
        Load the alignment points from a pickle file.
        """
        [self.ipf_points, self.sem_points] = io.load_prev_align_points()
        print(f"{len(self.ipf_points)} alignment points have been loaded.")

    def get_coord_before_tranformation(self, coord_after_transformation: list):
        """
        Get the coordinates of the selected points before the transformation.
        """
        [x, y] = coord_after_transformation
        original_coord = np.dot(
            self.inverse_transformation_matrix, np.array([[x], [y], [1]])
        )
        # Normalize the coordinates
        original_x = int(original_coord[0, 0] / original_coord[2, 0])
        original_y = int(original_coord[1, 0] / original_coord[2, 0])

        if (
            original_x > self.xmap.shape[1]
            or original_y > self.xmap.shape[0]
            or original_x < 0
            or original_y < 0
        ):
            Warning("Point is outside the EBSD map. Please confirm the input images.")
            return [-1, -1]
        else:
            return [original_x, original_y]

    def get_euler_from_sem_coord(self, sem_coord: list):
        """
        Get the Euler angles from the SEM coordinates.
        """
        [original_x, original_y] = self.get_coord_before_tranformation(
            [sem_coord[0], sem_coord[1]]
        )

        if (original_x == -1) or (original_y == -1):
            return [-1, -1, -1]
        else:
            [Eu1, Eu2, Eu3] = np.rad2deg(
                Rotation.to_euler(
                    self.xmap[int(original_y), int(original_x)].orientations
                )
            )[0]
            print(f"Euler angles at the image point: {[Eu1, Eu2, Eu3]}")
            return [Eu1, Eu2, Eu3]

    def get_phase_name(self, sem_coord: list):
        """
        Get the phase name from the SEM coordinates.
        """
        [original_x, original_y] = self.get_coord_before_tranformation(
            [sem_coord[0], sem_coord[1]]
        )

        if (original_x == -1) or (original_y == -1):
            Warning(f"Point is outside the EBSD map. Please confirm the input images.")
            return -1

        else:
            phase_name = self.xmap[int(original_y), int(original_x)].phases_in_data[:]
            print(f"Phase name at the image point: {phase_name}")
            return phase_name

    def interactive_blended_xmap(self, initial_coord: list):
        """
        Display the blended image and allow the user to click on the image to get the Euler angles.
        """
        self.coord_results = []

        [centre_x, centre_y] = initial_coord

        fig, ax3 = plt.subplots()
        ax3.imshow(self.ipf_warp_blended)
        ax3.plot(centre_x, centre_y, "r+", markersize=14)

        [original_x, original_y] = self.get_coord_before_tranformation(
            [centre_x, centre_y]
        )

        # initialise the lists to store the coordinates of the points of marker, clicked points in SEM image, corresponding points in IPF image, and Euler angles
        marker = [[centre_x, centre_y]]
        sem_coords = [[centre_x, centre_y]]
        ipf_coords = [[original_x, original_y]]
        euler3 = [self.get_euler_from_sem_coord([centre_x, centre_y])]
        phase_name = [self.get_phase_name([centre_x, centre_y])]

        def onclick(event):
            if event.dblclick:
                x, y = int(event.xdata), int(
                    event.ydata
                )  # Get the pixel coordinates from the click event
                plt.cla()

                ax3.imshow(self.ipf_warp_blended)

                [original_x, original_y] = self.get_coord_before_tranformation([x, y])

                sem_coords.append([x, y])
                ipf_coords.append([original_x, original_y])
                phase_name.append(self.get_phase_name([x, y]))

                [Eu1, Eu2, Eu3] = self.get_euler_from_sem_coord([x, y])

                marker.append(ax3.plot(x, y, "r+", markersize=14))
                if (Eu1 == -1) or (Eu2 == -1) or (Eu3 == -1):
                    ax3.set_title(
                        f"Clicked point is outside the EBSD map. Please select a point inside the map."
                    )
                else:
                    ax3.set_title(
                        f"IPF coordinate: {original_x}, {original_y},\nEuler angles: {Eu1:.2f}, {Eu2:.2f}, {Eu3:.2f}"
                    )
                # print(f"Euler angles: {Eu1:.2f}, {Eu2:.2f}, {Eu3:.2f}")
                euler3.append([Eu1, Eu2, Eu3])
                plt.draw()
                plt.axis("off")
                # print(len(marker))
                return sem_coords, ipf_coords, euler3, phase_name

        # Connect the mouse click event to the function
        cid = fig.canvas.mpl_connect("button_press_event", onclick)

        # Display the image and allow interactive selection
        plt.show()
        plt.draw()
        plt.axis("off")

        self.coord_results = [sem_coords, ipf_coords, euler3, phase_name]

        return sem_coords, ipf_coords, euler3, phase_name

    def interactive_rkp(
        self,
        ref_ECP_path,
        RKP_masterpattern,
        corr_angles,
        cam_length: float = 4,
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
        # st_rot = Rotation.from_axes_angles([0, 0, 1], -st_rot_angle, degrees=True)
        # st_tilt = Rotation.from_axes_angles([0, 1, 0], -st_tilt_angle, degrees=True)
        sem_coord = self.coord_results[0][-1]
        euler3 = self.coord_results[2][-1]
        phase = self.coord_results[3][-1]

        # this is required for oxford data, better be converted at an earlier stage to avoid manual conversion.
        fe_xtal_rotation = Rotation.from_euler(
            np.deg2rad(euler3)
        ) * Rotation.from_axes_angles([0, 0, 1], -np.pi / 2)

        fig, axes = plt.subplots(2, 2, figsize=[14, 10])

        # Plot the SEM image with the clicked point
        axes[0, 0].imshow(self.ipf_warp_blended)
        axes[0, 0].plot(sem_coord[0], sem_coord[1], "r+", markersize=14)
        axes[0, 0].axis("off")
        axes[0, 0].set_title("Overview SEM image with IPF Overlay", loc="center")

        # Plot the RKP overview with indexing
        axes[0, 1].cla()
        ref = ReciprocalLatticeVector(
            phase=phase, hkl=[[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]]
        )
        ref = ref.symmetrise().unique()
        hkl_sets = ref.get_hkl_sets()
        hkl_sets
        simulator = kp.simulations.KikuchiPatternSimulator(ref)

        sim_RKP_lowMag = rkp.get_sim_rkp(
            RKP_masterpattern,
            xtal_rotation=fe_xtal_rotation,
            st_rot_angle=0,
            st_tilt_angle=0,
            corr_angles=corr_angles,
            ref_ECP=ref_ECP_path,
            cam_length=0.6,
            RKP_shape=[1024, 1024],
        )
        sim_RKP_lowMag_pattern = np.squeeze(sim_RKP_lowMag.data)
        sim = simulator.on_detector(
            sim_RKP_lowMag.detector, sim_RKP_lowMag.xmap.rotations
        )
        axes[0, 1].imshow(sim_RKP_lowMag_pattern, cmap="gray")
        axes[0, 1].set_title("RKP Overview with indexing", loc="center")
        axes[0, 1].axis("off")

        lines, zone_axes, zone_axes_labels = sim.as_collections(
            zone_axes=True,
            zone_axes_labels=True,
            zone_axes_labels_kwargs=dict(fontsize=12),
        )
        axes[0, 1].add_collection(lines)
        axes[0, 1].add_collection(zone_axes)
        for label in zone_axes_labels:
            axes[0, 1].add_artist(label)

        rect = patches.Rectangle(
            (512 - (307 // 2), 512 - (307 // 2)),
            307,
            307,
            linewidth=2,
            edgecolor="royalblue",
            facecolor="none",
        )
        axes[0, 1].add_patch(rect)

        # Plot a second RKP with smaller angular range to show more detailed kikuchi band close to projection centre
        axes[1, 1].cla()
        sim_RKP_hiMag = rkp.get_sim_rkp(
            RKP_masterpattern,
            xtal_rotation=fe_xtal_rotation,
            st_rot_angle=0,
            st_tilt_angle=0,
            corr_angles=corr_angles,
            ref_ECP=ref_ECP_path,
            cam_length=2,
            RKP_shape=[1024, 1024],
        )
        sim_RKP_hiMag_pattern = np.squeeze(sim_RKP_hiMag.data)
        ecp_dim = list(sim_RKP_hiMag_pattern.shape)
        # get the BKP detector physcial dimensions
        [PCx_rkp, PCy_rkp, PCz_rkp] = sim_RKP_hiMag.detector.pc[0]
        [Ny, Nx] = sim_RKP_hiMag.detector.shape
        px_size_rkp = sim_RKP_hiMag.detector.px_size
        binning_rkp = sim_RKP_hiMag.detector.binning

        axes[1, 1].imshow(sim_RKP_hiMag_pattern, cmap="gray")
        # plot the centre marker for rkp prjection centre
        axes[1, 1].plot(
            PCx_rkp * Nx, PCy_rkp * Ny, "+", c="red", markersize=15, markeredgewidth=3
        )

        axes[1, 1].axis("off")
        axes[1, 1].set_title(
            "Click on RKP to find recommended stage movements", loc="center"
        )
        # ax2.set_title(f"Eular angle ({fe_euler_rotation.to_euler(degrees=True)}), required st rot {st_rot_target}, required st tilt {st_tilt_target}\n", fontsize=10)
        coords = []

        axes[1, 0].axis("off")

        def on_click2(event):
            if event.dblclick:
                # print(event.xdata, event.ydata)
                coords.append(event.xdata)
                coords.append(event.ydata)

                # plt.clf()
                axes[1, 0].cla()
                axes[1, 1].cla()

                axes[1, 1].imshow(sim_RKP_hiMag_pattern, cmap="gray")
                try:
                    x_pos = coords[-2]
                    y_pos = coords[-1]
                except:
                    x_pos = 0
                    y_pos = 0
                axes[1, 1].plot(
                    x_pos, y_pos, "+", c="yellow", markersize=15, markeredgewidth=3
                )
                axes[1, 1].plot(
                    PCx_rkp * Nx,
                    PCy_rkp * Ny,
                    "+",
                    c="red",
                    markersize=15,
                    markeredgewidth=3,
                )
                axes[1, 1].set_title(
                    "Click on RKP to find recommended stage movements", loc="center"
                )
                axes[1, 1].axis("off")

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
                    text = f"""
                    Calculation results:
                    
                    Pixel position: {int(x_pos)}, {int(y_pos)}
                    Virtual RKP detector pixel size: 10um
                    Virtual RKP detector camera length: {round(distance_l/1000,2)}mm
                    RKP resolution: [1024,1024]
                    Estimated SEM stage coordinates:
                    
                    Physical distance to PC on RKP detector:
                    x {round(distance_x,2)}um, y {round(distance_y,2)}um
                    
                    Stage Rot {round(math.degrees(azi_rkp),2)}\N{DEGREE SIGN}, 
                    Stage tilt {round(math.degrees(polar_rkp),2)}\N{DEGREE SIGN}
                    """
                elif stage_mode == "double-tilt":
                    theta_x_rkp = math.atan(distance_x / distance_l)
                    theta_y_rkp = math.atan(distance_y / distance_l)
                    test = f"""
                    Calculation results:
                    
                    Pixel position: {int(x_pos)}, {int(y_pos)}
                    Virtual RKP detector pixel size: 10um
                    Virtual RKP detector camera length: {round(distance_l/1000, 2)}mm
                    RKP resolution: [1024,1024]
                    Estimated SEM stage coordinates:
                    
                    Physical distance to PC on RKP detector:
                    x {round(distance_x,2)}um, y {round(distance_y,2)}um
                    
                    Tilt around Xs: {round(math.degrees(theta_x_rkp),2)}\N{DEGREE SIGN}, 
                    Tilt around Ys: {round(math.degrees(theta_y_rkp),2)}\N{DEGREE SIGN}
                    """

                axes[1, 0].axis("off")
                text_kwargs = dict(fontsize=14, ha="left", va="top", color="black")
                axes[1, 0].text(-0.1, 0.8, text, **text_kwargs)
                plt.draw()

        fig.canvas.mpl_connect("button_press_event", on_click2)
        plt.show()
        plt.draw()
        plt.axis("off")

        plt.tight_layout()
        fig.canvas.manager.window.move(0, 0)

        # return coords  # coordintes in x, y format
