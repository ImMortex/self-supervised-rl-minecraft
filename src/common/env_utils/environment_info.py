import re

from src.common.helpers.helpers import load_from_json_file


def get_minecraft_version_short():
    version = get_minecraft_version()
    return re.match("^\d{1,}\.\d{1,}", version).group(0)


def get_minecraft_version():
    minecraft_conf: dict = load_from_json_file("config/minecraft_conf.json")
    version = minecraft_conf["version"]
    return version

def get_application_name()->str:
    return "Minecraft " + get_minecraft_version()