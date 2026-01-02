import boto3
import os

s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET"]

s3.download_file(bucket, "models/latest/best.pt", "best.pt")
