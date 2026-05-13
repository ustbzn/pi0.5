import cv2
import numpy as np
from typing import Tuple, Dict
from scipy.interpolate import splprep, splev

class DrawTraj():
    def __init__(self):
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                              [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                              [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                    [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.coor2topdown = np.array([[1.0,  0.0,  0.0,  0.0], 
                                    [0.0, -1.0,  0.0,  0.0], 
                                    [0.0,  0.0, -1.0, 50.0], 
                                    [0.0,  0.0,  0.0,  1.0]])
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown 

    def draw_traj_bev(
        self, 
        traj: np.ndarray, 
        raw_img: np.ndarray, 
        canvas_size: Tuple[int, int] = (512, 512), 
        thickness: int = 3, 
        is_ego: bool = False, 
        hue_start: int = 120, 
        hue_end: int = 80
    ) -> np.ndarray:
        """
        Draws a bird's-eye view trajectory on an image with color gradient.

        Args:
            traj (np.ndarray): Array of trajectory points in world coordinates (N,2).
            raw_img (np.ndarray): Background image to draw on (3-channel RGB).
            canvas_size (tuple): Size of the output canvas (width, height).
            thickness (int): Line thickness for drawing.
            is_ego (bool): Whether this is the ego vehicle's trajectory.
            hue_start (int): Starting hue value for color gradient (0-180 for OpenCV).
            hue_end (int): Ending hue value for color gradient.

        Returns:
            np.ndarray: Image with the trajectory drawn (3-channel RGB).

        Note:
            - Uses cubic spline interpolation for smooth trajectory visualization.
            - Applies perspective transformation using self.coor2topdown matrix.
            - Implements HSV color gradient along trajectory length.
        """
        # Handle ego vehicle case (prepend origin point)
        if is_ego:
            line = np.concatenate([np.zeros((1,2)), traj], axis=0)
        else:
            line = traj
        
        img = raw_img.copy()        
    
        # Convert to homogeneous coordinates and apply perspective transform
        pts_4d = np.stack([line[:,0], line[:,1], np.zeros((line.shape[0])), np.ones((line.shape[0]))])
        pts_2d = (self.coor2topdown @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]  # Normalize x
        pts_2d[:, 1] /= pts_2d[:, 2]  # Normalize y
    
        # Filter points outside canvas
        mask = (
            (pts_2d[:, 0] > 0) & 
            (pts_2d[:, 0] < canvas_size[1]) & 
            (pts_2d[:, 1] > 0) & 
            (pts_2d[:, 1] < canvas_size[0])
        )
        if not mask.any():
            return img
        pts_2d = pts_2d[mask, 0:2]
    
        # Smooth trajectory with spline interpolation
        try:
            tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except:
            return img
    
        # Sample points along the smoothed curve
        unew = np.linspace(0, 1, 100)
        smoothed_pts = np.stack(splev(unew, tck)).astype(int).T
    
        # Draw gradient-colored line segments
        num_points = len(smoothed_pts)
        for i in range(num_points-1):
            # Calculate color gradient
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
            rgb_color_tuple = (float(rgb_color[0]), float(rgb_color[1]), float(rgb_color[2]))
        
            # Draw if within bounds
            if (
                smoothed_pts[i,0] > 0 and 
                smoothed_pts[i,0] < canvas_size[1] and 
                smoothed_pts[i,1] > 0 and 
                smoothed_pts[i,1] < canvas_size[0]
            ):
                cv2.line(
                    img,
                    (smoothed_pts[i,0], smoothed_pts[i,1]),
                    (smoothed_pts[i+1,0], smoothed_pts[i+1,1]),
                    color=rgb_color_tuple, 
                    thickness=thickness
                )   
            elif i == 0:
                break
            
        return img

    def draw_traj(
        self,
        traj: np.ndarray,
        raw_img: np.ndarray,
        canvas_size: Tuple[int, int] = (900, 1600),
        thickness: int = 3,
        is_ego: bool = True,
        hue_start: int = 120,
        hue_end: int = 80
    ) -> np.ndarray:
        """
        Draws a smooth trajectory on an image with perspective transformation and color gradient.

        Args:
            traj (np.ndarray): Array of trajectory points in 2D coordinates (N,2).
            raw_img (np.ndarray): Background image to draw on (3-channel RGB).
            canvas_size (Tuple[int, int]): Size of the output canvas (height, width).
            thickness (int): Thickness of the drawn trajectory line.
            is_ego (bool): Whether this is the ego vehicle's trajectory (adds starting point).
            hue_start (int): Starting hue value for color gradient (0-180).
            hue_end (int): Ending hue value for color gradient (0-180).

        Returns:
            np.ndarray: Image with the trajectory drawn (3-channel RGB).

        Note:
            - Uses perspective transformation from self.lidar2img['CAM_FRONT'].
            - Applies cubic spline interpolation for smooth trajectory.
            - Implements HSV color gradient along the trajectory.
            - For ego vehicle, adds a fixed starting point at (800, 900).
        """
        line = traj
        lidar2img_rt = self.lidar2img['CAM_FRONT']
        img = raw_img.copy()

        # Convert to homogeneous coordinates with fixed z=-1.84
        pts_4d = np.stack([
            line[:, 0],
            line[:, 1],
            np.ones((line.shape[0])) * -1.84,
            np.ones((line.shape[0]))
        ])

        # Apply perspective transformation and normalize
        pts_2d = (lidar2img_rt @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]  # Normalize x
        pts_2d[:, 1] /= pts_2d[:, 2]  # Normalize y

        # Filter points outside canvas bounds
        mask = (
            (pts_2d[:, 0] > 0) &
            (pts_2d[:, 0] < canvas_size[1]) &
            (pts_2d[:, 1] > 0) &
            (pts_2d[:, 1] < canvas_size[0])
        )
        if not mask.any():
            return img
        pts_2d = pts_2d[mask, 0:2]

        # Add fixed starting point for ego vehicle
        if is_ego:
            pts_2d = np.concatenate([np.array([[800, 900]]), pts_2d], axis=0)

        # Apply spline interpolation for smooth trajectory
        try:
            tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except:
            return img

        # Sample points along the smoothed curve
        unew = np.linspace(0, 1, 100)
        smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

        # Draw gradient-colored trajectory
        num_points = len(smoothed_pts)
        for i in range(num_points - 1):
            # Calculate color gradient
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
            rgb_color_tuple = (float(rgb_color[0]), float(rgb_color[1]), float(rgb_color[2]))

            # Draw line segment
            cv2.line(
                img,
                (smoothed_pts[i, 0], smoothed_pts[i, 1]),
                (smoothed_pts[i+1, 0], smoothed_pts[i+1, 1]),
                color=rgb_color_tuple,
                thickness=thickness
            )

        return img