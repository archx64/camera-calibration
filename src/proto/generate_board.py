import cv2
import numpy as np

def create_a4_charuco():
    # --- 1. Configuration ---
    # A4 Dimensions in mm
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297
    
    # margins, printers need ~10mm edge
    MARGIN_MM = 15 
    
    # grid configuration (5x7 fits A4 portrait well)
    SQUARES_X = 5
    SQUARES_Y = 7
    
    # dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # --- 2. Calculate Sizes ---
    # available width for the board
    usable_width = A4_WIDTH_MM - (2 * MARGIN_MM)
    usable_height = A4_HEIGHT_MM - (2 * MARGIN_MM)
    
    # Calculate max square side length that fits in the usable width
    max_sq_width = usable_width / SQUARES_X
    max_sq_height = usable_height / SQUARES_Y
    
    # Use the smaller of the two to ensure it fits both ways
    square_length_mm = min(max_sq_width, max_sq_height)
    
    # Round down to nearest millimeter for easier measuring later
    square_length_mm = int(square_length_mm) 
    
    # Define Marker size (usually 75% of square)
    marker_length_mm = int(square_length_mm * 0.75)

    print(f"--- BOARD CONFIGURATION ---")
    print(f"Squares: {SQUARES_X} x {SQUARES_Y}")
    print(f"Square Length: {square_length_mm} mm")
    print(f"Marker Length: {marker_length_mm} mm")
    print(f"Total Board Size: {SQUARES_X*square_length_mm}mm x {SQUARES_Y*square_length_mm}mm")
    print("---------------------------")

    # --- 3. Convert to Pixels for 300 DPI ---
    # Formula: pixels = (mm / 25.4) * DPI
    DPI = 300
    
    # board physical size in pixels
    board_width_px = int((SQUARES_X * square_length_mm / 25.4) * DPI)
    board_height_px = int((SQUARES_Y * square_length_mm / 25.4) * DPI)
    
    # margins in pixels
    margin_px = int((MARGIN_MM / 25.4) * DPI)
    
    # total image size
    img_width_px = board_width_px + 2 * margin_px
    img_height_px = board_height_px + 2 * margin_px
    
    # Generate Board
    # we must convert millimeters to meters for the OpenCV function
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), 
        square_length_mm / 1000.0, 
        marker_length_mm / 1000.0, 
        aruco_dict
    )
    
    # generate the board image (just the pattern)
    pattern_img = board.generateImage((board_width_px, board_height_px), marginSize=0)
    
    # create a white A4 canvas
    final_img = np.ones((img_height_px, img_width_px), dtype=np.uint8) * 255
    
    # paste the pattern in the center in order to account for margins
    final_img[margin_px:margin_px+board_height_px, margin_px:margin_px+board_width_px] = pattern_img

    # Save
    filename = f"charuco_A4_{square_length_mm}mm_squares.png"
    cv2.imwrite(filename, final_img)
    print(f"Saved as '{filename}'.")
    print(f"IMPORTANT: Update your calibration code to use SQUARE_LEN = {square_length_mm / 1000.0}")

create_a4_charuco()