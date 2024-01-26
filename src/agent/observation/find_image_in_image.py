import time

import cv2
import numpy as np


def find_image_in_image(input_img: np.ndarray, target_img: np.ndarray, threshold: float = 0.98,
                        debug: bool = False) -> (bool, (int, int), (int, int)):
    """
    tutorial: https://learncodebygaming.com/blog/opencv-object-detection-in-games-python-tutorial-1 (27.06.2023)
    :@param threshold: confidence if image was found
    :@param input_img: BGR Image in which an image is to be found
    :@param target_img: BGR Image that is sought
    :@param debug: to visualize the result
    :@return
    """
    start = time.time()

    # convert BGRA to BGR
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGRA2BGR)

    #
    # There are 6 comparison methods to choose from:
    # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
    # You can see the differences at a glance here:
    # https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
    # Note that the values are inverted for TM_SQDIFF and TM_SQDIFF_NORMED
    result = cv2.matchTemplate(input_img, target_img, cv2.TM_CCOEFF_NORMED)
    # You can view the result of matchTemplate() like this:
    # cv.imshow('Result', result)
    # cv.waitKey()
    # If you want to save this result to a file, you'll need to normalize the result array
    # from 0..1 to 0..255, see:
    # https://stackoverflow.com/questions/35719480/opencv-black-image-after-matchtemplate
    # cv.imwrite('result_CCOEFF_NORMED.jpg', result * 255)
    # Get the best match position from the match result.
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # The max location will contain the upper left corner pixel position for the area
    # that most closely matches our target image. The max value gives an indication
    # of how similar that find is to the original target, where 1 is perfect and -1
    # is exact opposite.

    target_found: bool = False
    top_left = (0, 0)
    bottom_right = (0, 0)
    if max_val >= threshold:
        target_found = True

    if debug:
        # Calculate the bottom right corner of the rectangle to draw
        top_left = max_loc
        bottom_right = (top_left[0] + target_img.shape[1], top_left[1] + target_img.shape[0])

        # Draw a rectangle on our screenshot to highlight where we found the target.
        # The line color can be set as an RGB tuple
        output_img = input_img.copy()
        cv2.rectangle(output_img, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
        print("Best match top left position:", str(max_loc))
        print("Best match confidence", max_val, "using threshold", threshold)
        print("top_left", top_left)
        print("bottom_right", bottom_right)
        print("find_image_in_image", str(time.time() - start))
        print("found", target_found)
        cv2.imshow('Result', output_img)
        cv2.imshow('Target', target_img)
        cv2.waitKey(3)

    return target_found, top_left, bottom_right
