import os
import subprocess

# Define a helper function to run shell commands
def run_command(cmd):
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")

# Step 1: Initialize DVC if not already done
if not os.path.exists(".dvc"):
    run_command("dvc init")
else:
    print("DVC is already initialized, skipping `dvc init`.")

# Step 2: Set up a valid remote storage directory
remote_path = r"C:\Users\Hasan\Desktop\EDA\Week3_Challenge_Project\RemoteStorage"
if not os.path.exists(remote_path):
    os.makedirs(remote_path)
run_command(f"dvc remote add -d localstorage {remote_path} --force")

# Step 3: Track the dataset with DVC
dataset = "MachineLearningRating_v3.txt"
if os.path.exists(dataset):
    run_command(f"dvc add {dataset}")
    run_command(f"git add {dataset}.dvc .gitignore")
    run_command('git commit -m "Track dataset with DVC"')
else:
    raise FileNotFoundError(f"The dataset file '{dataset}' does not exist.")

# Step 4: Push data to remote storage
run_command("dvc push")




