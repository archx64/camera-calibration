import cv2, os
from copy import deepcopy

# CAM1_SOURCE = 1
# CAM2_SOURCE = 2
WEB_CAM = False
CAM1_SOURCE = 'rtsp://admin:csimAIT5706@192.168.6.101:554/Streaming/Channels/101/'
CAM2_SOURCE = 'rtsp://admin:csimAIT5706@192.168.6.100:554/Streaming/Channels/101/'

BASE_DIR = "calibration_data"
FOLDER1 = os.path.join(BASE_DIR, "cam1")
FOLDER2 = os.path.join(BASE_DIR, "cam2")




def setup_folders():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists(FOLDER1):
        os.makedirs(FOLDER1)
    if not os.path.exists(FOLDER2):
        os.makedirs(FOLDER2)
    print(f"folders created inside {BASE_DIR}")


def main():
    setup_folders()

    # cap1 = cv2.VideoCapture(CAM1_SOURCE, cv2.CAP_DSHOW)
    # cap2 = cv2.VideoCapture(CAM2_SOURCE, cv2.CAP_DSHOW)

    cap1 = cv2.VideoCapture(CAM1_SOURCE)
    cap2 = cv2.VideoCapture(CAM2_SOURCE)

    # desired_width = 640
    # desired_height = 360

    # cap1.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    # cap2.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)

    # cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    # cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    if not cap1.isOpened() or not cap2.isOpened():
        print(
            f"camera 1 status: {cap1.isOpened()} | camera 2 status: {cap2.isOpened()}"
        )
        return

    print("\nControls:")
    print("'s' -> save synchronized frame")
    print("'q' -> Quit")

    count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("error: lost signal from a camera")
            break

        # resize the frame to fit on the monitor
        display_h = 480
        aspect_ratio = frame1.shape[1] / frame1.shape[0]
        display_w = int(display_h * aspect_ratio)

        frame1_copy = deepcopy(frame1)
        frame2_copy = deepcopy(frame2)

        show1 = cv2.resize(frame1_copy, (display_w, display_h))
        show2 = cv2.resize(frame2_copy, (display_w, display_h))

        if WEB_CAM:
            show1 = cv2.flip(show1, 1)
            show2 = cv2.flip(show2, 1)

        combined = cv2.hconcat([show1, show2])

        cv2.putText(
            combined,
            f"saved: {count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Stereo Data Collector", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            filename = f"frame_{count:03d}.jpg"

            p1 = os.path.join(FOLDER1, filename)
            p2 = os.path.join(FOLDER2, filename)

            cv2.imwrite(p1, frame1)
            cv2.imwrite(p2, frame2)

            count += 1

            print(f'saved pair {count}: {filename}')

            white_flash = combined.copy()
            white_flash[:] = (255, 255, 255)
            cv2.imshow("Stereo Data Collector", white_flash)
            cv2.waitKey(50)

        elif key == ord("q"):
            break

    cap1.release()
    cap2.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
