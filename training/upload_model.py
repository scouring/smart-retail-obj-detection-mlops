import boto3
import os
import datetime

bucket = os.environ["S3_BUCKET"]
s3 = boto3.client("s3")

timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

s3.upload_file(
    "runs/detect/train/weights/best.pt",
    bucket,
    f"models/{timestamp}/best.pt"
)

s3.upload_file(
    "runs/detect/train/weights/best.pt",
    bucket,
    "models/latest/best.pt"
)

print("Model uploaded:", timestamp)
