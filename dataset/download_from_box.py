import os
import requests

# Function to download a single file
def download_file(file_url, output_dir, overwrite=False):
    file_name = os.path.basename(file_url)
    output_path = os.path.join(output_dir, file_name)
    if not overwrite and os.path.exists(output_path):
        print(f"Skipping {file_name} (already exists)")
        return
    print(f"Downloading {file_name}...")
    response = requests.get(file_url)
    with open(output_path, "wb") as f:
        f.write(response.content)

# Function to download a Box folder recursively
def download_box_folder(folder_url, output_dir, overwrite=False):
    response = requests.get(folder_url)
    folder_data = response.json()
    entries = folder_data["item_collection"]["entries"]
    for entry in entries:
        item_type = entry["type"]
        item_name = entry["name"]
        if item_type == "file":
            file_url = entry["shared_link"]["download_url"]
            download_file(file_url, output_dir, overwrite)
        elif item_type == "folder":
            folder_id = entry["id"]
            subfolder_url = f"https://api.box.com/2.0/folders/{folder_id}/items"
            subfolder_output_dir = os.path.join(output_dir, item_name)
            os.makedirs(subfolder_output_dir, exist_ok=True)
            download_box_folder(subfolder_url, subfolder_output_dir, overwrite)

# Main script
if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download files from Box folder")
    parser.add_argument("--box_folder_url", default="https://berkeley.box.com/s/y9obxoojqyy2zskh5kp613wid3zsue0h", help="URL of the Box folder")
    parser.add_argument("--output_directory", default="nE2024_data", help="Output directory for downloaded files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    box_folder_url = args.box_folder_url
    output_directory = args.output_directory
    overwrite = args.overwrite

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Download the Box folder recursively
    download_box_folder(box_folder_url, output_directory, overwrite)

    print("Download complete!")
