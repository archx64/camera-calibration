import os, glob, cv2, json
import numpy as np
from utils import *

PATH_CAM1 = "calibration_data/cam1/*.jpg"
PATH_CAM2 = "calibration_data/cam2/*.jpg"

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARES_LENGTH, MARKER_LENGTH, aruco_dict
)

CAMERAS = [
    {
        "name": "cam1",
        "path": os.path.join(IMAGES_DIR, "cam1/*.jpg"),
        "is_reference": True,
    }
]

# image_dirs = dict()

for i in range(2, CAMERA_COUNT + 1):
    CAMERAS.append(
        {
            "name": f"cam{i}",
            "path": os.path.join(IMAGES_DIR, f"cam{i}/*.jpg"),
            "is_reference": False,
        }
    )
    # image_dirs[f"cam{i+1}"] = os.path.join(IMAGES_DIR, f"cam{i+1}/*.jpg")

print(json.dumps(CAMERAS, indent=4))


def detect_corners(cam_config):
    """Detect ChArUco corners from a single camera"""

    name = cam_config["name"]
    path = cam_config["path"]
    print(f"scanning path for {name}...")

    images = sorted(glob.glob(path))

    if not images:
        print(ERROR + f"no images found in {path}")
        return None

    data_dict = dict()
    all_corners = list()
    all_ids = list()
    img_shape = None

    for frame in images:
        img = cv2.imread(frame)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            ret, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=board
            )

            if ret > 6:
                all_corners.append(char_corners)
                all_ids.append(char_ids)
                key = os.path.basename(frame)
                data_dict[key] = (char_corners, char_ids)

    print(INFO + f"[{name}] found {len(all_corners)} valid frames")

    return {
        "data_dict": data_dict,
        "all_corners": all_corners,
        "all_ids": all_ids,
        "shape": img_shape,
    }


def main():
    results = dict()

    for cam in CAMERAS:
        res = detect_corners(cam)
        if res is None:
            exit()

        results[cam["name"]] = res
        # print(DEBUG + f'data type of results: {type(results)}')
        # print(DEBUG + f"keys in res: {results.keys()}")
        # print(DEBUG + f"keys in cam1: {results['cam1']}")

    # let's calibrate intrinsics individually
    intrinsics = dict()
    print(INFO + "\nphase1: intrinsic calibration")

    for cam in CAMERAS:
        name = cam["name"]
        res_name = results[name]

        # print(DEBUG + f"keys in res: {res.keys()}")
        # print(DEBUG + f"type of res['shape']: {type(res['shape'])}, value of res['shape']: {res['shape']}")
        # print(DEBUG + f"keys in res['shape']: {res['shape'].keys()}")

        print(f"solving intrinsics for {name}")

        inputK = np.array([])
        inputD = np.array([])

        # ret, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
        #     charucoCorners=res["all_corners"],
        #     ids=res["all_ids"],
        #     board=board,
        #     imageSize=res["shape"],
        #     cameraMatrix=inputK,
        #     distCoeffs=inputD,
        #     rvecs=None,
        #     tvecs=None
        # )

        ret, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=res_name["all_corners"],
            charucoIds=res_name["all_ids"],
            board=board,
            imageSize=res_name["shape"],
            cameraMatrix=inputK,
            distCoeffs=inputD,
        )

        print(f"RMSE: {ret:.4f}")
        intrinsics[name] = {"K": K, "D": D, "shape": res["shape"], "rmse": ret}
    
    # let's calibrate extrinsics
    print('\nphase 2: extrinsic stereo calibration')

    # identify reference camera
    ref_cam = next((c for c in CAMERAS if c['is_reference']), None)

    if not ref_cam:
        print(ERROR + "Error: No camera marked as 'is_reference': 'True'")
        exit()
    
    ref_name = ref_cam['name']
    ref_data = results[ref_name]['data_dict']
    ref_intrinsics = intrinsics[ref_name]

    final_output = {
        'reference_camera' : ref_name,
        'camera': {}
    }

    # add reference camera to output (identity matrix)
    final_output['cameras'][ref_name]  = {
        'K': ref_intrinsics['K'],
        'D': ref_intrinsics['D'],
        'R': np.eye(3),
        'T': np.zeros((3,1)),
        'rmse': ref_intrinsics['rmse']
    }

    # iterate over satellites (peripherical camers)
    for cam in CAMERAS:
        target_name = cam['name']

        if target_name == ref_name: # skip master camera
            continue

        print(INFO + f'syncing {ref_name} to {target_name}')
        target_data = results[target_name]['data_dict']
        target_intrinsics = intrinsics[target_name]



if __name__ == "__main__":
    # detect_corners(CAMERAS[1])
    main()
