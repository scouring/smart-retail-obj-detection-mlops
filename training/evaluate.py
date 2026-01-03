import os
import json
import boto3
from ultralytics import YOLO
from datetime import datetime, timezone

BUCKET = os.environ["S3_BUCKET"]

# Load the model
model = YOLO("runs/detect/train/weights/best.pt")

# Run validation
metrics = model.val()

# Save metrics to S3
metrics_dict = metrics.results_dict

with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

s3 = boto3.client("s3")
s3.upload_file("metrics.json", BUCKET, "metrics/latest/metrics.json")

# Save metrics to S3
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

# Upload all files in runs/detect/val/
val_dir = "runs/detect/val"
for file in os.listdir(val_dir):
    local_path = os.path.join(val_dir, file)
    s3.upload_file(local_path, BUCKET, f"metrics/{timestamp}/{file}")

print("Metrics uploaded to S3:", timestamp)

