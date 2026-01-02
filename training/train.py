import os
import boto3
import torch
from ultralytics import YOLO

# Environment variables
BUCKET = os.environ["S3_BUCKET"]
DATA_PREFIX = os.environ.get("S3_DATA_PREFIX", "datasets/latest/")
LOCAL_DATA_DIR = "data"

# -------- Download dataset from S3 --------
s3 = boto3.client("s3")

def download_s3_folder(bucket, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            target = obj["Key"].replace(prefix, local_dir + "/")
            if target.endswith("/"):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            s3.download_file(bucket, obj["Key"], target)

download_s3_folder(BUCKET, DATA_PREFIX, LOCAL_DATA_DIR)

# -------- Train YOLO --------
device = "cpu"

model = YOLO("yolov8n.pt")

model.train(
    data="training/data.yaml",
    epochs=50,
    imgsz=640,
    device=device
)
