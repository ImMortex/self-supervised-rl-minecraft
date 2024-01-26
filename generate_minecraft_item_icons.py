import re
import shutil
from difflib import SequenceMatcher
from os import listdir, path, makedirs
from os.path import isfile, join
from pathlib import Path

import minecraft_data

from src.common.env_utils.environment_info import get_minecraft_version_short


def generate_minecraft_item_icons():
    global match
    short_version = get_minecraft_version_short()
    mc_data: minecraft_data.mod = minecraft_data(short_version)
    items_list: [] = mc_data.items_list
    downloaded_icons_dir = "./tmp/minecraft_icons"
    final_icons_dir = "agent_assets/icon_classifier_data/minecraft_icons"
    if not path.exists(final_icons_dir):
        makedirs(final_icons_dir)
    icon_file_names = [f for f in listdir(downloaded_icons_dir) if
                       isfile(join(downloaded_icons_dir, f)) and f.endswith(".png")]
    matches = []
    for item in items_list:
        name: str = item["name"]
        id: str = item["id"]
        new_filename = str(id) + "_" + name + ".png"
        a_cleaned = name.lower()

        results = []
        for filename in icon_file_names:
            filename_no_suffix = str(Path(filename).with_suffix(""))
            filename_no_suffix = re.sub("\d", "", filename_no_suffix)
            filename_no_suffix = filename_no_suffix.replace("_", " ")
            filename_no_suffix = filename_no_suffix.strip()
            b_cleaned = filename_no_suffix.lower()

            # results.append((fuzz.partial_token_set_ratio(a, b), a))
            results.append((SequenceMatcher(None, a_cleaned, b_cleaned).ratio(), filename))

        result = max(results)
        match = [new_filename, result[1]]
        matches.append(match)
        print(match)
        try:
            # copy file
            shutil.copy(downloaded_icons_dir + "/" + match[1], final_icons_dir + "/" + match[0])
        except Exception as e:
            print(e)


generate_minecraft_item_icons()
