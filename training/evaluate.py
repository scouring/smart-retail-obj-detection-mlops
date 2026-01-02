import os
import json
import boto3
from ultralytics import YOLO

BUCKET = os.environ["S3_BUCKET"]

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val()

metrics_dict = metrics.results_dict

with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

s3 = boto3.client("s3")
s3.upload_file("metrics.json", BUCKET, "metrics/latest/metrics.json")

print("Metrics uploaded to S3")

