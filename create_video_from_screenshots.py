
import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join

from config.train_config import get_train_config

train_config = get_train_config()

step_length = 1 / train_config["t_per_second"]
steps_per_epoch = train_config["steps_per_epoch"]
def create_video(img_array=None, output_file_path='output_video.avi'):
    if img_array is None:
        img_array = []

    if len(img_array) == 0:
        return
    height, width, layers = img_array[0].shape
    size = (width, height)

    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def convert_png_to_gif(png_paths, gif_path, duration=200, loop=0):
    images = []

    # Open each PNG image and append it to the images list
    for png_path in png_paths:
        image = Image.open(png_path)
        images.append(image)

    # Save the images as a GIF with the specified duration and loop
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )




dir_path = "./tmp/screenshots_to_video/"

# Get all paths of PNG images in the specified directory
png_paths = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(".png")]

sort_objects = []
for path in png_paths:
    id = int(path.split(".png")[0].split("timestep_")[1])
    sort_objects.append({"id": id, "path": path})

new_list = sorted(sort_objects, key=lambda x: x["id"], reverse=False)

sorted_png_paths = []
for o in new_list:
    sorted_png_paths.append(o["path"])

images = [Image.open(fn) for fn in sorted_png_paths]
"""
images = []
for path in png_paths:
    image = cv2.imread(path)
    images.append(image)
"""


out_dir = "./tmp/video_output/"
filename = "output_gif.gif"
frame_one = images[0]
frame_one.save(out_dir + filename, format="GIF", append_images=images,
               save_all=True, duration=len(images)*step_length, loop=0)

#convert_png_to_gif(png_paths=png_paths, gif_path= directory_path + "output.gif", duration=128*4, loop=0)

#create_video(img_array=images)
