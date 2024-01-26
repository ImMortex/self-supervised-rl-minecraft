import logging
import time


def countdown(sec: int, optional_text: str = None, cancel_condition_function = None):
    for i in list(range(sec))[::-1]:
        if optional_text is None:
            logging.info(i + 1)
        else:
            logging.info(str(i + 1) + " " + optional_text)
        time.sleep(1)
        if cancel_condition_function is not None and cancel_condition_function():
            break
