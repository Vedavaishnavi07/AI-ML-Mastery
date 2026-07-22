import os
import tensorflow as tf

dataset = "data/dataset"

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

bad_files = []

for root, dirs, files in os.walk(dataset):
    for file in files:
        path = os.path.join(root, file)

        if not file.lower().endswith(extensions):
            continue

        try:
            image = tf.io.read_file(path)
            tf.io.decode_image(image)
        except Exception:
            bad_files.append(path)

if bad_files:
    print("\nBad images found:\n")
    for file in bad_files:
        print(file)
else:
    print("\nAll images are valid!")