import cv2
import imutils
import numpy as np
from imutils import contours

from src.agent.observation.compare_images import mse_imgs

bank_of_digits: dict = None  # must be calculated only once


def get_digit_mc_digit_image_tresh(img: np.ndarray) -> np.ndarray:
    """
    This function is used to convert a img that contains a white minecraft digit into a black white image.

    @param img: BGR or BGRA image
    @return: Black white image. White font.
    """
    gray_img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_img, 245, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh


def get_digit_mc_digit_image_green_color(img: np.ndarray) -> np.ndarray:
    """
    This function is used to convert a img that contains a green minecraft digit into a black white image.

    @param img: BGR or BGRA image
    @return: Black white image. White font.
    """
    # hsv used to better delineate colors
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # green hsv mask
    mask = cv2.inRange(img_hsv, np.array([40, 0, 0]), np.array([70, 255, 255]))
    detected = cv2.bitwise_and(img, img, mask=mask)  # green number with optional green background
    gray_img: np.ndarray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)

    clearance = 10
    threshold = gray_img.max() - clearance  # threshold adjusted to brightness
    thresh = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # filename = "./tmp/digitGreenThresh" + str(time.time()) + ".png"
    # cv2.imwrite(filename, thresh)
    return thresh


def create_bank_of_digits_0_to_9() -> dict:
    """
    Creates comparison examples for digit mse classifier from given files
    @return: dict
    """
    bank_of_digits: dict = {}
    # load digit examples from 0-9
    for i in range(10):
        filename = "agent_assets/minecraft_digits/" + str(i) + ".png"
        image = cv2.imread(filename, 1)
        # cv2.imshow("image " + str(i), images)
        # cv2.waitKey(-1)

        thresh = get_digit_mc_digit_image_tresh(image)
        cnt = get_digit_contours(thresh)
        (x, y, w, h) = cv2.boundingRect(cnt[0])  # get only first contour, only one digit expected
        bank_of_digits[i] = thresh[y:y + h, x:x + w]

    return bank_of_digits


def get_bank_of_digits():
    """
    @return: Comparison examples for digit mse classifier
    """
    global bank_of_digits
    if bank_of_digits is None:
        bank_of_digits = create_bank_of_digits_0_to_9()
    return bank_of_digits


def get_digit_contours(thresh: np.ndarray, digit_width=(18, 22), digit_height=(28, 32)) -> []:
    """
    Returns bounding boxes around recognized digits in the given image.

    @param thresh: Black white image. White font.
    @return: List of bounding rectangles
    """
    cnt = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = imutils.grab_contours(cnt)
    digit_contours = []
    # loop over the digit area candidates
    for c in cnt:
        (x, y, w, h) = cv2.boundingRect(c)  # compute the bounding box of the contour
        if (w >= digit_width[0] and w <= digit_width[1]) and (h >= digit_height[0] and h <= digit_height[
            1]):  # if the contour is sufficiently large, it must be a digit
            digit_contours.append(c)
    # sort the contours from left-to-right, then initialize the actual digits themselves
    if len(digit_contours) > 0:
        digit_contours = contours.sort_contours(digit_contours, method="left-to-right")[0]
    return digit_contours


def classify_single_digit_mse(digit: np.ndarray, bank_of_digits: dict, best_score: float = 10000) -> int:
    """
    Classifies the single digit (from 0 to 9) in the given image.
    @param digit: Black white image that contains only a digit with white font
    @param bank_of_digits: Comparison examples for digit mse classifier
    @param best_score: max allowed mse
    @return: int from 0 to 9
    """
    # code based on https://stackoverflow.com/questions/52083129/digit-recognizing-using-opencv (20.06.2023)
    # cv2.imshow("image", digit)
    # cv2.waitKey(-1)
    matched_digit = 0  # Default
    for number in bank_of_digits:
        digit = cv2.resize(digit, (bank_of_digits[number].shape[1], bank_of_digits[number].shape[0]),
                           interpolation=cv2.INTER_AREA)
        score = mse_imgs(digit, bank_of_digits[number])
        # print("Score for number " + str(number) +" is: "+ str(np.round(score,2)) )
        if score < best_score:  # If we find better match
            best_score = score  # Set highest score yet
            matched_digit = number  # Set best match number
    # print("Best match: " + str(matched_digit))
    return matched_digit
