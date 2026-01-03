import os
import subprocess

print("Starting training...")
subprocess.run(["python", "training/train.py"], check=True)

print("Evaluating model...")
subprocess.run(["python", "training/evaluate.py"], check=True)

print("Uploading model...")
subprocess.run(["python", "training/upload_model.py"], check=True)

print("Pipeline complete.")
