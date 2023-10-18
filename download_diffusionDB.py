from urllib.request import urlretrieve
import pandas as pd

# Download the parquet table
table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(table_url, 'metadata.parquet')

# Extract every 100th row from downloaded file
prompts = pd.read_parquet('metadata.parquet', columns=['image_name', 'prompt'])
every_100th_row = prompts.iloc[::100]
output_csv_file = "output_every_100th_row.csv"
every_100th_row.to_csv(output_csv_file, index=False)