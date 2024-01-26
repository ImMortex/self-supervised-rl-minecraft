import os
from os import listdir
from os.path import isfile, join

from PIL import Image

from src.agent.observation.observe_inventory.libs.observe_inventory_slots import get_inventory_slots_test_mapping_image

dir_path = "./tests/inventory_img_original/"
file_names = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(".png")]

output_dir_path = "tmp/testInventoryMapping/inventory_img_marked_mapping/"

if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

for filename in file_names:
    img: Image = Image.open(dir_path + filename)
    img = get_inventory_slots_test_mapping_image(img)
    img.save(output_dir_path + filename)

print("Results saved to " + output_dir_path)
