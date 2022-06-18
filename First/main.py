import cv2
import os
import re

def merge_channels(input_dir, output_dir):
    images = []
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file in os.listdir(input_dir):
        images.append(file)

        if len(images) == 3:
            cvs = [cv2.split(cv2.imread(os.path.join(input_dir, image))) for image in images]

            merged = cv2.merge([item[i] for i, item in enumerate(cvs)])
            cv2.imwrite(os.path.join(output_dir, re.sub("_b", "", images[0])), merged)
            images = []


merge_channels("../dataset/data/", "./merged")
