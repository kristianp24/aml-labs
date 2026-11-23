import os
import urllib.request
import zipfile

download_dir = "/content/download"
os.makedirs(download_dir, exist_ok=True)

zip_path = os.path.join(download_dir, "tiny-imagenet-200.zip")
extract_dir = os.path.join(download_dir, "tiny-imagenet")
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

# 1. Download
print("Downloading Tiny-ImageNet...")
urllib.request.urlretrieve(url, zip_path)

# 2. Extract silently
print("Extracting...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

# 3. Clean up zip
os.remove(zip_path)

print(f"Dataset ready at: {os.path.join(extract_dir, 'tiny-imagenet-200')}")