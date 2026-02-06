import os
from PIL import Image
import tensorflow as tf

dataset_path = "dataset"

bad_files = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)

        try:
            
            img = Image.open(file_path)
            img.verify()

            
            img_raw = tf.io.read_file(file_path)
            tf.image.decode_image(img_raw)

        except:
            print("Removing corrupt file:", file_path)
            os.remove(file_path)
            bad_files += 1

print("Total removed:", bad_files)
