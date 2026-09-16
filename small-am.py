from huggingface_hub import snapshot_download
import os

# Your specific path
target_path = r'C:\Users\Belayneh\Downloads\MedicalChatbot\model'

# Create the directory if it doesn't exist
os.makedirs(target_path, exist_ok=True)

print(f"Downloading model to: {target_path}...")

# Download the model
snapshot_download(
    repo_id="b1n1yam/whisper-small-am",
    local_dir=target_path,
    local_dir_use_symlinks=False  # This ensures actual files are copied, not shortcuts
)

print("Download complete!")
