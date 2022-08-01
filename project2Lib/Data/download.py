from pathlib import Path
import requests
import zipfile

def download_data(small = True, replace_num = True, data_dir = Path("./data")):

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Switch case for deciding to load the small or large dataset
    size = "20k" if small else "200k"

    # Switch case for replacing num with @
    replace = "_numbers_replaced_with_at_sign" if replace_num else ""

    # Load datasets directly from Git-Repo of Franck-Dernoncourt
    # The original repo can be found under the following URL:
    # https://github.com/Franck-Dernoncourt/pubmed-rct
    for prefix in ["train", "dev", "test"]:

        filename = prefix + ".txt"
        local_path = data_dir.joinpath(filename)
        base_url = f"https://github.com/Franck-Dernoncourt/pubmed-rct/raw/master" \
                   f"/PubMed_{size}_RCT{replace}/"

        # Only download data if necessery
        if not local_path.exists():

            # Try to download dataset as text file directly
            req_txt = requests.get(base_url + filename)
            if req_txt.ok:
                local_path.open("wb+").write(req_txt.content)

            # Otherwise the dataset is stored as a zip-file
            else:
                req_zip = requests.get(base_url + prefix + ".zip")
                
                # If this request fails, then raise an error
                if not req_zip.ok:
                    raise Exception(f"the following dataset is unavailable: {prefix}")

                # Store zipped content in temporary compressed file
                tmp_file = data_dir.joinpath(prefix + ".zip")
                tmp_file.open("wb+").write(req_zip.content)

                # Unzip data
                with zipfile.ZipFile(tmp_file, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                
                # Delete temp file
                tmp_file.unlink()
