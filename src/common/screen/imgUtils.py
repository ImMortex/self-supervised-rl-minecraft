import cv2

def resize(screenshot, dest_w, dest_h):
    return cv2.resize(screenshot, (dest_w, dest_h), interpolation=cv2.INTER_AREA)


def process_img_gray(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img
