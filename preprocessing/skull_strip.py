import subprocess
import os

def run_synthstrip(input_path, output_path):
    print("Running Skull Stripping")
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    model_path = "../training/synthstrip.pt"

    cmd = [
        "nipreps-synthstrip",
        "-i", input_path,
        "-o", output_path,
        "--model", model_path
    ]
    subprocess.run(cmd, check=True)