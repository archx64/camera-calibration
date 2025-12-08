import numpy as np
import json
from .utils import ERROR

def convert_to_mmpose_format(npz_path, output_path):
    try:
        data = np.load(npz_path)
    except FileNotFoundError as e:
        print(ERROR + str(e))
        return
    
    cam_names = sorted(list(set([k.split('_')[0] for k in data.files])))

    print(f'found cameras: {cam_names}')

    camera_dict = {}

    for name in cam_names:
        # load raw matrices
        K = data[f"{name}_K"] # intrinsic matrix (3x3)
        D = data[f"{name}_D"] # distortion coefficients
        R = data[f"{name}_R"] # rotation matrix (3x3) as list
        T = data[f"{name}_T"] # translation vector (3x1) as list (in millimeters usually, but meters is fine if consistent)

        # opencv returns D as (1, 5) or (5, 1). flatten it to a list
        dist_coeffs = D.flatten().tolist()

        # opencv calibration in meters
        # Human3.6M uses millimeters
        T_mm = (T*1000).flatten().tolist()

        camera_dict[name] = {
            'K': K.tolist(),
            'R': R.tolist(),
            'T': T_mm,
            'dist_coeffs': dist_coeffs,
            'img_width': 640,
            'img_height': 360, 
        }

    # save to json
    with open(output_path, 'w') as f:
        json.dump(camera_dict, f, indent=4)
    
    print(f'successfully converted to {output_path}')

if __name__ == '__main__':
    convert_to_mmpose_format(npz_path='multicam_calibration.npz', output_path='multicam_calibration.json')

