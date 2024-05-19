import requests
import os
import gzip
import pandas as pd

"""Downloads a file from a given URL to the specified directory."""

#Smaller fashion dataset: https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz

url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz"
output_directory = "/scratch/cgriu/AML_dataset/"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

file_name = url.split('/')[-1]
download_path = os.path.join(output_directory, file_name)

response = requests.get(url, stream=True, verify=False)
if response.status_code == 200:
    with open(download_path, 'wb') as f:
        f.write(response.raw.read())
    print(f"Download successful.")

    if file_name.endswith('.gz'):
        output_file_name = file_name[:-3]
        output_path = os.path.join(output_directory, output_file_name)

        with gzip.open(download_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print("Decompression successful.")

        os.remove(download_path)  # Remove the compressed file
    else:
        print("The file downloaded is not gzip.")

else:
    print("Failed!!")

csv_output_path = output_path.replace('.json', '.csv')

print(f"Converting {os.path.basename(output_path)} to CSV...")
df = pd.read_json(output_path, lines=True)
df.to_csv(csv_output_path, index=False)
print(f"Converted to CSV successfully at {csv_output_path}.")