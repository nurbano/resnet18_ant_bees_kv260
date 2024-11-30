import os
import requests
import zipfile
import io

# Check if the directory exists
if not os.path.exists("./dataset/hymenoptera_data"):
    # Download the zip file
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Save the zip file to disk
    zip_filename = "hymenoptera_data.zip"
    with open(zip_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Unzip the file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall("./dataset")

    # Delete the zip file
    os.remove(zip_filename)
else:
    print("Directory 'hymenoptera_data' already exists.")