import time

import cv2

from src.agent.observation.agent_make_screenshot import agent_make_screenshot
from src.common.countdown import countdown

countdown(5)
screenshot = agent_make_screenshot()
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
filename = "./tmp/screenshot " + str(time.time()) + ".png"
cv2.imwrite(filename, screenshot)
