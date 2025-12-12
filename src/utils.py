from colorama import Back, Fore, Style, init

init(autoreset=True)

ERROR = Fore.LIGHTRED_EX + Back.BLACK + Style.BRIGHT
HEAD = Fore.LIGHTGREEN_EX + Back.BLACK + Style.NORMAL
BODY = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.NORMAL

SUCCESS = Fore.LIGHTGREEN_EX + Back.BLACK + Style.BRIGHT
INFO = Fore.LIGHTBLUE_EX + Back.BLACK + Style.BRIGHT
DEBUG = Fore.LIGHTYELLOW_EX + Back.BLACK + Style.BRIGHT
WARNING = Fore.BLACK + Back.YELLOW + Style.NORMAL

TARGET_PAPER = 'A0'

CAMERA_COUNT = 2

# ISO A paper sizes in millimeters
PAPER_SIZES = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A2": (420, 594),
    "A1": (594, 841),
    "A0": (841, 1189),
}

PAPER_CONFIGS = {
    'A4': 0.036, #36mm
    'A3': 0.053, #53mm
    'A2': 0.078, # 78mm
    'A1': 0.112, # 112mm
    'A0': 0.162, # 0.162mm
}

if TARGET_PAPER  not in PAPER_CONFIGS:
    print(ERROR + f'you must select paper from {PAPER_CONFIGS}')
    exit()

SQUARES_LENGTH = PAPER_CONFIGS[TARGET_PAPER]
MARKER_LENGTH = SQUARES_LENGTH * 0.75

SQUARES_X = 5
SQUARES_Y = 7
MARGIN_MM = 15  # 15 mm is safe for most printers
DPI = 300

IMAGES_DIR = 'calibration_data'

# 25.4 millimeters in 1 inch
# pixels = (mm/25.4)*DPI
def board_pixel(dimension, sq_len_mm):
    pixels = int((dimension * sq_len_mm / 25.4) * DPI)
    return pixels


def total_pixel(dimension):
    pixels = int((dimension / 25.4) * DPI)
    return pixels

def offset_pixel(total, board):
    pixels = (total - board) // 2
    return pixels
