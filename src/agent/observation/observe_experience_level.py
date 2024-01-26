import cv2
import numpy as np

from src.agent.observation.digit import get_digit_contours, classify_single_digit_mse, \
    get_bank_of_digits, get_digit_mc_digit_image_green_color
from src.common.helpers.helpers import load_from_json_file

exp_level_digits: dict = load_from_json_file("config/experience_level_conf.json")


def observe_exp_level(screenshot: np.ndarray) -> int:
    """
    Feature extraction of the experience level value of a minecraft player from the given screenshot.
    @param screenshot: full minecraft screen BGR or RGB
    @return: experience bar value value from 0 to 100
    """
    # crop image (keep only important pixels of the exp bar), keep original unchanged
    cropped_img: np.ndarray = screenshot[exp_level_digits["top_left"][1]:exp_level_digits["bottom_right"][1],
                              exp_level_digits["top_left"][0]:exp_level_digits["bottom_right"][0]]
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # using RGB

    # original code: https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/ (19.06.2023)

    digit = 0  # default
    thresh = get_digit_mc_digit_image_green_color(cropped_img)  # white font, black background
    # cv2.imshow("Output", thresh)
    # cv2.waitKey(-1)

    #cv2.imwrite("./tmp/digits/" + str(time.time()) + ".png", thresh)

    digit_contours = get_digit_contours(thresh)
    digits = []

    # loop over each of the digits
    for c in digit_contours:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)

        # debug
        output = cropped_img.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.imshow("Output", output)
        # cv2.waitKey(-1)

        digit_img = thresh[y:y + h, x:x + w]
        # cv2.imshow("Output", digit_img)
        # cv2.waitKey(-1)
        classified_digit = classify_single_digit_mse(digit_img, get_bank_of_digits())
        digits.append(classified_digit)

    if len(digits) == 1:
        digit = digits[0]
    elif len(digits) >= 2:
        digit = ""

        for d in digits:
            digit += str(d)

        digit = int(digit)
    return digit
