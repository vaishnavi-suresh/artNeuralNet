# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile

api= KaggleApi()
api.authenticate()

# Set the path to the file you'd like to load

# Load the latest version

api.dataset_download_files(
    "ikarus777/best-artworks-of-all-time",
    path = './files'
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

df = pd.read_csv('./files/artists.csv')

with zipfile.ZipFile("./files/best-artworks-of-all-time.zip", 'r') as zip_ref:
    zip_ref.extractall("./files")


print("First 5 records:", df.head())
