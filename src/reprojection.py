import json
import cv2
import numpy as np

def validate_mmpose_json(json_path):
    with open(json_path, 'r') as f:
        cams = json.load(f)

    # define a 3D point in world space in the frame of master camera
    # let's pick a point 2 meters in front of Cam 1
    # X=0, Y=0, Z=2000mm since we converted to millimeters
    point_3d = np.array([0.0, 0.0, 2000.0]).reshape(3, 1)

    print(f"Testing Projection of 3D Point: {point_3d.flatten()} (mm)")

    # create dummy images, black canvas to draw on
    # or load real images if you have them: cv2.imread('cam1/frame_000.jpg')
    canvas_h, canvas_w = 360, 640
    # canvas = cv2.imread('calibration_data/cam1/frame_000.jpg')
    
    for name, params in cams.items():
        K = np.array(params['K'])
        D = np.array(params['dist_coeffs'])
        R = np.array(params['R'])
        T = np.array(params['T']).reshape(3, 1)
        
        # Standard Projection Equation:
        # P_cam = R * P_world + T
        cam_point = R @ point_3d + T
        
        # project to pixels
        # let's use cv2.projectPoints to handle K and D automatically
        # note: cv2.projectPoints expects 3D points, rvec (Rodrigues), tvec, K, D
        
        # convert R back to Rodrigues vector for the function
        rvec, _ = cv2.Rodrigues(R)
        
        # T is already translation vector
        img_points, _ = cv2.projectPoints(point_3d.T, rvec, T, K, D)
        
        x, y = int(img_points[0][0][0]), int(img_points[0][0][1])
        
        print(f"  {name}: projected to pixel ({x}, {y})")

        # visualization
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.circle(canvas, (x, y), 20, (0, 255, 0), -1) # green dot
        cv2.putText(canvas, f"{name} Projection", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(f"validation - {name}", canvas)
    
    print("press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    validate_mmpose_json('multicam_calibration.json')

