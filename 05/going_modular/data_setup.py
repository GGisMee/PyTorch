import importLib
import os
import requests
import zipfile
from pathlib import Path

# Setup basic files
data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'

# Create folder for dataset
image_path.mkdir(parents=True, exist_ok=True)

# Create zip file with data
https_address = r'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip'
importLib.import_from_github(https=https_address, directory=data_path)

with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")