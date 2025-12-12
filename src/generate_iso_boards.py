import os, cv2
import numpy as np
from .utils import *


print(f"data type: {type(cv2.aruco.DICT_6X6_250)} value: {cv2.aruco.DICT_6X6_250}")


def generate_iso_boards():

    if not os.path.exists("boards"):
        os.mkdir("boards")

    print(
        HEAD
        + f"{'PAPER': <6} | {'SQUARE SIZE (mm)': <18} | {'SQUARE SIZE (m)': <18} | {'MARKER SIZE (mm)': <18}"
    )
    print("-" * 72)

    for name, (w_mm, h_mm) in PAPER_SIZES.items():

        # calculate printable area
        usable_w = w_mm - (2 * MARGIN_MM)
        usable_h = h_mm - (2 * MARGIN_MM)
        # print(INFO + f"printable area ({name}): {usable_w}mm x {usable_h}mm")

        # calculate square size
        # let's use smaller dimension to ensure fit, round down to integer for easier measuring
        sq_size_x = usable_w / SQUARES_X
        sq_size_y = usable_h / SQUARES_Y
        sq_len_mm = int(min(sq_size_x, sq_size_y))
        marker_len_mm = int(sq_len_mm * 0.75)

        print(
            BODY
            + f"{name:<6} | {sq_len_mm: <18} | {sq_len_mm/1000.0: <18} | {marker_len_mm: <18}"
        )

        # let's create board object
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard(
            size=(SQUARES_X, SQUARES_Y),
            squareLength=sq_len_mm / 1000,
            markerLength=marker_len_mm / 1000,
            dictionary=aruco_dict,
        )

        # calculate pixel dimensions for 300 DPI
        # pixels = (mm/25.4)*DPI

        # board_w_px = board_pixel(SQUARES_X, sq_len_mm)
        # board_h_px = board_pixel(SQUARES_Y, sq_len_mm)
        board_w_px = int((SQUARES_X * sq_len_mm / 25.4) * DPI)
        board_h_px = int((SQUARES_Y * sq_len_mm / 25.4) * DPI)

        # total_w_px = total_pixel(w_mm)
        # total_h_px = total_pixel(h_mm)

        total_w_px = int((w_mm/25.4)*DPI)
        total_h_px = int((h_mm/25.4 * DPI))

        # generate boad
        pattern = board.generateImage((board_w_px, board_h_px))

        # white background
        canvas = np.ones((total_h_px, total_w_px), dtype=np.uint8) * 255

        # calculate offsets to center charuco
        # x_offset = offset_pixel(total_w_px, board_w_px)
        # y_offset = offset_pixel(total_h_px, board_h_px)

        x_offset = (total_w_px - board_w_px) // 2
        y_offset = (total_h_px - board_h_px) // 2

        canvas[y_offset : y_offset + board_h_px, x_offset : x_offset + board_w_px] = (
            pattern
        )

        filename = f"boards/charuco_{name}_{sq_len_mm}mm.png"
        cv2.imwrite(img=canvas, filename=filename)


if __name__ == "__main__":
    generate_iso_boards()
