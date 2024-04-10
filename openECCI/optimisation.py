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

from openECCI.util import normalize, rotate_image_around_point
from openECCI import io, rkp
import numpy as np
from orix.quaternion import Rotation
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os


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
        I1 = I1 + np.dot( image1[ii,:], np.conj(image2[ii,:]) )
        I2 = I2 + np.dot( image1[ii,:], np.conj(image1[ii,:]) )
        I3 = I3 + np.dot( image2[ii,:], np.conj(image2[ii,:]) )

    MAC = np.abs(I1)**2 / (I2*I3)
    return MAC

class orientation_calibration():
    def __init__(self,
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
        self.initial_guess = {"tiltX_corr_angle": self.corr_angles[0],
                              "tiltY_corr_angle": self.corr_angles[1],
                              "tiltZ_corr_angle": self.corr_angles[2], 
                              "PCz": self.cam_length}
        
        print(f"Orientation Calibration Object created using the following parameters: \
              \nInitial guess corrections: {self.initial_guess},\
              \nReference ECP: {self.reference_ECP_path},\
              \nMaster Pattern: {self.master_pattern}")
        
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
            exp_ECP = plt.imread(self.reference_ECP_path)[:ecp_shape[1], :]
        else: exp_ECP = plt.imread(self.reference_ECP_path)
        
        si_average_euler_rotation = self.Si_xtal

        sim_rkp = rkp.get_sim_rkp(RKP_masterpattern = self.master_pattern, 
                              xtal_rotation = si_average_euler_rotation, 
                              st_rot_angle=0,
                              st_tilt_angle=0,
                              corr_angles=parameter_list[0:3],
                              ref_ECP=self.reference_ECP_path,
                              cam_length=parameter_list[3],
                              RKP_shape=ecp_shape)

        MAC = 1-modal_assurance_criterion(exp_ECP, np.squeeze(sim_rkp.data))
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
            exp_ECP = plt.imread(self.reference_ECP_path)[:ecp_shape[1], :]
        else: exp_ECP = plt.imread(self.reference_ECP_path)
        
        si_average_euler_rotation = self.Si_xtal

        sim_rkp = rkp.get_sim_rkp(RKP_masterpattern = self.master_pattern, 
                              xtal_rotation = si_average_euler_rotation, 
                              st_rot_angle=0,
                              st_tilt_angle=0,
                              corr_angles=parameter_list[0:3],
                              ref_ECP=self.reference_ECP_path,
                              cam_length=parameter_list[3],
                              RKP_shape=ecp_shape)
        
        NDP = 1-np.dot(normalize(exp_ECP.flatten()), normalize(np.squeeze(sim_rkp.data).flatten()))
        
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
        
        initial_guess = [self.initial_guess['tiltX_corr_angle'], 
                         self.initial_guess['tiltY_corr_angle'], 
                         self.initial_guess['tiltZ_corr_angle'], 
                         self.initial_guess['PCz']]
        if method == "MAC":
            res = minimize(self._get_minus_MAC, initial_guess, method='Nelder-Mead', tol=1e-6, options={'disp': True} )
            result = res.x
            print("Rotation calibration optimized using MAC method")
            print(f"tiltX_corr_angle: {result[0]}, tiltY_corr_angle: {result[1]}, tiltZ_corr_angle: {result[2]} PCz: {result[3]}")
            print(f"MAC: {1-self._get_minus_MAC(result)}")
            return {"tiltX_corr_angle": result[0],
                    "tiltY_corr_angle": result[1],
                    "tiltZ_corr_angle": result[2], 
                    "PCz": result[3]}
        
        elif method == "NDP":
            res = minimize(self._get_minus_NDP, initial_guess, method='Nelder-Mead', tol=1e-6, options={'disp': True})
            result = res.x
            print("Rotation calibration optimized using NDP method")
            print(f"tiltX_corr_angle: {result[0]}, tiltY_corr_angle: {result[1]}, tiltZ_corr_angle: {result[2]} PCz: {result[3]}")
            print(f"NDP: {1-self._get_minus_NDP(result)}")
            return {"tiltX_corr_angle": result[0],
                    "tiltY_corr_angle": result[1],
                    "tiltZ_corr_angle": result[2], 
                    "PCz": result[3]}

        else:
            raise ValueError("Current supported methods are 'MAC' and 'NDP', please choose the supported method.")
        
        
class convergence_angle_measurement():
    def __init__(self, 
                 aperture_image_path: str,
                 ):
        """
        A process to measure the electron beam convergence (semi) angle 
        """
        self.img_metadata = io.get_sem_metadata(aperture_image_path)
        img_resolution = self.img_metadata['resolution']
        
        img = cv2.imread(aperture_image_path, cv2.IMREAD_ANYDEPTH)
        img_gray = img[:img_resolution[1], :img_resolution[0]]  
        self.image = img_gray
        self.img_resolution = img_resolution
        _ , self.filename = os.path.split(aperture_image_path)
        
    def __repr__(self):
        return f"Convergence Angle Measurement Object: {self.Pt_aperture_image_path}"
    
    def find_centroid(self, 
                      threshold: float,
                      plot: bool = True):
        """
        find the centroid of the aperture image without any rotation.
        """
        ret,thresh = cv2.threshold(self.image , threshold, np.max(self.image), cv2.THRESH_BINARY_INV)
        
        M = cv2.moments(thresh, False)
        # calculate x,y coordinate of circle center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        if plot:
            plt.figure()
            plt.imshow(self.image, cmap='gray')
            plt.imshow(thresh, alpha=0.3)
            plt.scatter(cX,cY,c="r" )
            plt.show()
        
        self.original_centroid = [cX, cY]
        # reduce the profile range by 10 pixels to avoid the edge of the image
        self.profile_range = np.min(self._get_distance_to_edge([cX, cY]))-10
        
        return [cX, cY]
    
    def _get_edge_profile(self, 
                          aperture_image: np.ndarray,
                          centroid: list,
                          filter_sigma: float = 5,
                          plot: bool = False):
        """
        Get the edge profile from  aperture image along the horizontal line from the centroid to the right
        """
        profile = aperture_image[int(centroid[1]),int(centroid[0]):(int(centroid[0]) + self.profile_range)]
        normalized_profile = normalize(profile)
        gradient_1st = gaussian_filter(np.gradient(normalized_profile), sigma=filter_sigma)
        gradient_2nd = gaussian_filter(np.gradient(gradient_1st), sigma=filter_sigma)
        normalized_gradient_2nd = normalize(gradient_2nd)   
        
        x = np.arange(0, len(normalized_profile)) * self.img_metadata['pixel_size'] * 1e6
        
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(x, normalized_profile)
            plt.plot(x, normalize(gradient_1st), label='1st derivative', color='orange')
            plt.plot(x, normalize(gradient_2nd), label='2nd derivative', color='green')
            plt.xlabel('Distance (um)')
            plt.ylabel('Intensity (a.u.)')
            plt.legend()
            
            plt.scatter(x[np.argmax(gradient_2nd)], np.max(normalized_gradient_2nd), color='r')
            plt.scatter(x[np.argmin(gradient_2nd)], np.min(normalized_gradient_2nd), color='r')
            plt.scatter(x[np.argmax(gradient_2nd)], normalized_profile[np.argmax(gradient_2nd)], color='b')
            plt.scatter(x[np.argmin(gradient_2nd)], normalized_profile[np.argmin(gradient_2nd)], color='b')
            
            plt.show()
        
        return normalized_profile, gradient_1st, gradient_2nd
    
    def _get_distance_to_edge(self, point_coord: list):
        """
        Get the distance from the centroid to the edge of the initial unrotated image in the four directions.
        """
        [cX, cY] = point_coord
        return [cX, cY, self.img_resolution[0]-cX, self.img_resolution[1]-cY]
    
    def get_profile_from_angle(self,
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
        rotated_img, new_centroid = rotate_image_around_point(self.image, centroid, -angle)
        profile, gradient_1st, gradient_2nd = self._get_edge_profile(rotated_img, new_centroid, filter_sigma, plot=plot)
        
        return profile, gradient_1st, gradient_2nd 
    
    def _conv_angle_from_profile(self, gradient_2nd: np.ndarray):
        """
        Compute the convergence angle from the 2nd derivative of the edge profile.        
        """
        probe_diameter = (np.argmin(gradient_2nd)- np.argmax(gradient_2nd)) * self.img_metadata['pixel_size']
        tan_alpha = (probe_diameter/2) / (self.img_metadata['stage_z'] - self.img_metadata['WD'])
        conv_half_angle = np.arctan(tan_alpha) * 1000 # in mrad
        
        return conv_half_angle, probe_diameter
    
    def compute(self,
                angle_step: float,
                filter_sigma: float = 5,
                plot: bool = True
                ):
        """ 
        Compute the gradient of the edge profile of the aperture image along different angles.
        """
        steps = np.arange(0, 359, angle_step)
        
        for index, value in enumerate(tqdm(steps)):
            profile, gradient_1st, gradient_2nd  = self.get_profile_from_angle(angle=value, 
                                                                               filter_sigma=filter_sigma,
                                                                               plot=False)
            conv_half_angle, probe_diameter = self._conv_angle_from_profile(gradient_2nd)
            
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
            fig, axes = plt.subplots(2,2, figsize=[14,10])
            
            x = np.arange(0, profiles.shape[1]) * self.img_metadata['pixel_size'] * 1e6
            
            [cX, cY] = self.original_centroid
            axes[0,0].imshow(self.image, cmap='gray')
            axes[0,0].scatter(cX, cY, c="r" )
            axes[0,0].set_title('Defocused Aperture Image')
            
            axes[0,1].plot(x, profiles[0,:], label='Profile')
            axes[0,1].plot(x, normalize(gradients_1st[0,:]), '-', label='1st derivative')
            axes[0,1].plot(x, normalize(gradients_2nd[0,:]), '-', label='2nd derivative')
            axes[0,1].scatter(x[np.argmax(gradient_2nd)], np.max(normalize(gradients_2nd[0,:])), color='r')
            axes[0,1].scatter(x[np.argmin(gradient_2nd)], np.min(normalize(gradients_2nd[0,:])), color='r')
            axes[0,1].scatter(x[np.argmax(gradient_2nd)], profiles[0,:][np.argmax(gradient_2nd)], color='b')
            axes[0,1].scatter(x[np.argmin(gradient_2nd)], profiles[0,:][np.argmin(gradient_2nd)], color='b')
            axes[0,1].set_xlabel('Distance (um)')
            axes[0,1].set_ylabel('Intensity (a.u.)')
            axes[0,1].legend()
            
            axes[1,0].plot(steps, probe_diameters*1e6, 'x', color='b', label='Diameter of diverged probe disk')
            # axes[1,0].set_title(f'Diameter of diverged probe disk vs. angle')
            axes[1,0].set_xlabel('Angle of measurement (deg)')
            axes[1,0].set_ylabel('Diverged probe disk diameter (um)')
            ax2 = axes[1,0].twinx()
            ax2.plot(steps, conv_half_angles, 'o', color='r', label='Convergence angle')
            ax2.set_ylabel('Convergence angle (mrad)')
            ax2.set_ylim([0, 15])
            axes[1,0].legend()
            ax2.legend()
            
            # Add legend
            points1, label1 = axes[1,0].get_legend_handles_labels()
            points2, label2 = ax2.get_legend_handles_labels()
            ax2.legend(points1 + points2, label1 + label2)
            
            # filename = f"""
            # HV{high_voltage}_curr{beam_current}_mode{lens_mode}_Z{position["z"]}_WD{WD}
            # """

            # Aperture Dia.:      {aperture_img_metadata['ApertureDiameter']*1e6} um
            # Beam Current:       {aperture_img_metadata['BeamCurrent']*1e9} nA
            # LensMode:           {aperture_img_metadata['LensMode']}

            text = f"""
            Filename: {self.filename}
            High voltage: {self.img_metadata['HV']} kV
            StageZ: {self.img_metadata['stage_z']*1e3} mm
            WD: {self.img_metadata['WD']*1e3} mm
            Average probe diameter: {round(np.mean(probe_diameters)*1e6, 3)} um
            Average convergence half angle: {round(np.mean(conv_half_angles), 3) } mrad
            Stdev convergence half angle: {round(np.std(conv_half_angles), 3) } mrad
            convergence angle range : {round(np.degrees(np.mean(conv_half_angles) * 2 /1000), 3)} deg
            """
            axes[1,1].axis('off')
            axes[1,1].text(0, 0, text, fontsize=12)
        
        
        self.angles = steps
        self.profiles = profiles
        self.gradients_1st = gradients_1st
        self.gradients_2nd = gradients_2nd
        self.conv_half_angles = conv_half_angles
        self.probe_diameters = probe_diameters
    
