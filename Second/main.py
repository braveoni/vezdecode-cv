import cv2
import os
import pandas as pd
import torch


model = torch.hub.load("ultralytics/yolov5", "yolov5s")


def find_car(input_dir, output_cars="output.csv"):
    df = {}

    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            result = model(os.path.join(input_dir, file))
            
            df[file] =  True if "car" in result.pandas().xyxy[0]["name"].tolist() else False

    df = pd.DataFrame(
        {
            "filename": list(df.keys()),
            "value": list(df.values())
        }
    )

    df.to_csv(output_cars, index=False)


find_car("./merged/")
