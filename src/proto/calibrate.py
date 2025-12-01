import cv2
import numpy as np
import glob
import os


# use the same values from the A4 charuco generation script
SQUARES_X = 5  
SQUARES_Y = 7  

# square and market length in meters
SQUARE_LENGTH = 0.036 
MARKER_LENGTH = 0.027 # Usually 75% of square_length

# path to images captured from two cameras
PATH_CAM1 = 'calibration_data/cam1/*.jpg'
PATH_CAM2 = 'calibration_data/cam2/*.jpg'

# initialize the ChArUco board dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), 
    SQUARE_LENGTH, 
    MARKER_LENGTH, 
    aruco_dict
)

def detect_charuco_corners(images_path, cam_name):
    """
    Detects corners and returns them in a dictionary for easy syncing.
    Returns: 
        data_dict: { filename_key: (corners, ids) }
        image_shape: (width, height)
        all_corners: list for intrinsic calibration
        all_ids: list for intrinsic calibration
    """
    print(f"[{cam_name}] Scanning images...")
    images = sorted(glob.glob(images_path))
    
    if not images:
        print(f"Error: No images found in {images_path}")
        exit()

    all_corners = []
    all_ids = []
    data_dict = {} # key: filename suffix (e.g., "frame_01")
    
    image_shape = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None: image_shape = gray.shape[::-1]

        # detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        # interpolate ChArUco corners
        if len(corners) > 0:
            ret, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            
            # at least 6 corners are needed for a good solve
            if ret > 6:
                all_corners.append(char_corners)
                all_ids.append(char_ids)
                
                # store by filename to sync with the other camera later
                # let's assume filenames are like "cam1/frame_10.jpg" and "cam2/frame_10.jpg"
                # use the simple filename "frame_10.jpg" as the key
                key = os.path.basename(fname)
                data_dict[key] = (char_corners, char_ids)

                vis_img = img.copy()

                cv2.aruco.drawDetectedMarkers(vis_img, corners)

                cv2.aruco.drawDetectedCornersCharuco(vis_img, char_corners, char_ids, (0,255,0))

                cv2.imshow(f'Calibration View - {cam_name}', vis_img)\
                
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()


    cv2.destroyWindow(f'Calibration View - {cam_name}')
    print(f"[{cam_name}] Found {len(all_corners)} valid frames.")
    return data_dict, image_shape, all_corners, all_ids

# detect Features
data1, shape1, corners_list1, ids_list1 = detect_charuco_corners(PATH_CAM1, "Cam 1")
data2, shape2, corners_list2, ids_list2 = detect_charuco_corners(PATH_CAM2, "Cam 2")

# calibrate individual cameras (Intrinsics)
print("\n--- Calibrating Intrinsics ---")
print("Solving Cam 1...")
ret1, K1, D1, _, _ = cv2.aruco.calibrateCameraCharuco(
    corners_list1, ids_list1, board, shape1, None, None
)
print(f"Cam 1 RMSE: {ret1:.4f} px")

print("Solving Cam 2...")
ret2, K2, D2, _, _ = cv2.aruco.calibrateCameraCharuco(
    corners_list2, ids_list2, board, shape2, None, None
)
print(f"Cam 2 RMSE: {ret2:.4f} px")

# prepare data for stereo calibration
print("\n--- Preparing Stereo Data ---")
stereo_obj_points = []
stereo_img_points1 = []
stereo_img_points2 = []

# find keys (filenames) that exist in BOTH cameras
common_keys = sorted(list(set(data1.keys()) & set(data2.keys())))
print(f"Found {len(common_keys)} synchronized frames.")

for key in common_keys:
    c1, id1 = data1[key]
    c2, id2 = data2[key]
    
    # find the intersection of IDs (points visible in BOTH cameras for this frame)
    id1_vals = id1.flatten()
    id2_vals = id2.flatten()
    common_ids = np.intersect1d(id1_vals, id2_vals)
    
    # need enough common points to establish geometry
    if len(common_ids) < 6: 
        continue
        
    # get 3D object points for these specific IDs
    # the board knows where "ID 5" is in 3D space
    obj_pts_all = board.getChessboardCorners()
    obj_pts = obj_pts_all[common_ids]
    
    # get 2D image points
    # filter the original arrays to keep only the common IDs
    mask1 = np.isin(id1_vals, common_ids)
    mask2 = np.isin(id2_vals, common_ids)
    
    pts1 = c1[mask1]
    pts2 = c2[mask2]
    
    stereo_obj_points.append(obj_pts)
    stereo_img_points1.append(pts1)
    stereo_img_points2.append(pts2)

print(f"Used {len(stereo_obj_points)} frames for final stereo solve.")

# Stereo Calibration
print("\n--- Running Stereo Calibration ---")
# intrinsics (K, D) are fixed because they are solved individually.
# this stabilizes the result significantly.
flags = cv2.CALIB_FIX_INTRINSIC

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    stereo_obj_points,
    stereo_img_points1,
    stereo_img_points2,
    K1, D1,
    K2, D2,
    shape1,
    criteria=criteria,
    flags=flags
)

print(f"FINAL STEREO RMSE: {ret_stereo:.4f}")
print(f"Translation Vector (T):\n{T}")
print(f"Rotation Matrix (R):\n{R}")

# 5. Save
np.savez('stereo_params.npz', K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F)
print("Calibration saved to 'stereo_params.npz'")