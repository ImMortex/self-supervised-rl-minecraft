import cv2
import numpy as np

from src.common.screen.screenshotUtils import get_screenshot


def agent_make_screenshot() -> np.ndarray:
    screenshot: np.ndarray = get_screenshot(dest_w=1920, dest_h=1080)  # the pov view of the mc player
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)  # not RGB
    return screenshot