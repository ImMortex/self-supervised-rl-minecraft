import logging
import traceback

import pygetwindow


def force_stop_if_app_not_found(application_name) -> bool:
    force_stop = False
    try:
        title: str = pygetwindow.getActiveWindowTitle()
        if application_name not in title:
            force_stop = True
            logging.warning(
                "Agent force stop. No window active with name " + application_name)
    except Exception as e:
        logging.error("Error while validating application name")
        logging.error(e)
        traceback.print_exc()
    return force_stop
