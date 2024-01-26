window_height = 0
window_width = 0

from screeninfo import get_monitors

for m in get_monitors():

    if m.is_primary:
        window_height = m.height
        window_width = m.width


def get_main_screen_origin_x() -> int:
    return 0


def get_main_screen_origin_y() -> int:
    return 0


def get_main_screen_height() -> int:
    return window_height


def get_main_screen_width() -> int:
    return window_width
