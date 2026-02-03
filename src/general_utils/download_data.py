import os
import requests
import tarfile
import zipfile
import gzip
from tqdm import tqdm

from src import PROJECT_ROOT

# Use absolute path based on script location to ensure consistent data directory
# This will always point to the project's data folder regardless of where the script is called from
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "external")
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")


def download_datasets(datasets: dict[str, str]):
    for name, url in datasets.items():
        filepath = os.path.join(DATA_DIR, name)
        if not os.path.exists(filepath):
            print(f"Donwloading {name}...")

            # Stream download with progress bar
            response = requests.get(url, stream=True, verify=False)
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(filepath, "wb") as f,
                tqdm(
                    desc=name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"{name} downloaded to {filepath}")
        else:
            print(f"{name} already present.")


def extract_files(datasets: dict[str, str]):
    """Extract downloaded compressed files"""
    for name in datasets.keys():
        filepath = os.path.join(DATA_DIR, name)
        if os.path.exists(filepath):
            if name.endswith((".tar.gz", ".tgz")):
                base_name = name[:-7] if name.endswith(".tar.gz") else name[:-4]
                extract_dir = os.path.join(DATA_DIR, base_name)
                if not os.path.exists(extract_dir):
                    print(f"Extraction de {name}...")
                    with tarfile.open(filepath, "r:gz") as tar:
                        members = tar.getmembers()
                        with tqdm(
                            total=len(members), desc=f"Extracting {name}"
                        ) as pbar:
                            for member in members:
                                tar.extract(member, DATA_DIR)
                                pbar.update(1)
                    print(f"{name} extracted in {extract_dir}")
                else:
                    print(f"{name} already extracted.")

            elif name.endswith(".csv.gz"):
                extract_path = os.path.join(DATA_DIR, name.replace(".gz", ""))
                if not os.path.exists(extract_path):
                    print(f"Extraction de {name}...")
                    with gzip.open(filepath, "rb") as f_in:
                        with open(extract_path, "wb") as f_out:
                            # Get compressed file size for progress bar
                            compressed_size = os.path.getsize(filepath)

                            with tqdm(
                                total=compressed_size,
                                desc=f"Extracting {name}",
                                unit="B",
                                unit_scale=True,
                            ) as pbar:
                                while True:
                                    chunk = f_in.read(8192)
                                    if not chunk:
                                        break
                                    f_out.write(chunk)
                                    # Update progress based on compressed bytes read
                                    pbar.update(len(chunk))
                    print(f"{name} extracted to {extract_path}")
                else:
                    print(f"{name} already extracted.")

            elif name.endswith(".zip"):
                extract_dir = os.path.join(DATA_DIR, name.replace(".zip", ""))
                if not os.path.exists(extract_dir):
                    print(f"Extracting {name}...")
                    with zipfile.ZipFile(filepath, "r") as z:
                        members = z.namelist()
                        with tqdm(
                            total=len(members), desc=f"Extracting {name}"
                        ) as pbar:
                            for member in members:
                                z.extract(member, DATA_DIR)
                                pbar.update(1)
                    print(f"{name} extracted in {extract_dir}")
                else:
                    print(f"{name} already extracted.")


def delete_files(datasets: dict[str, str]):
    for name in datasets.keys():
        if not name.endswith((".tar.gz", ".tgz", ".zip", ".csv.gz")):
            continue
        filepath = os.path.join(DATA_DIR, name)
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    _datasets = {
        "LP_PDBBind.csv": "https://github.com/THGLab/LP-PDBBind/raw/refs/heads/master/dataset/LP_PDBBind.csv",
    }
    download_datasets(_datasets)
    extract_files(_datasets)
    delete_files(_datasets)
