import argparse
from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download

REPO_ID = "l-lt/LaSOT"
SAVE_DIR = Path("data/LaSOT")


def download_category(category_name: str, unzip: bool = True):
    global SAVE_DIR
    SAVE_DIR_CATEGORY = SAVE_DIR / category_name
    SAVE_DIR_CATEGORY.mkdir(parents=True, exist_ok=True)

    zip_name = f"{category_name}.zip"

    print(f"Downloading {zip_name}...")

    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=zip_name,
        local_dir=SAVE_DIR_CATEGORY,
    )

    print(f"Downloaded: {zip_path}")

    if unzip:
        print(f"Extracting {zip_name}...")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(SAVE_DIR_CATEGORY)

        print(f"Done: {SAVE_DIR_CATEGORY}/{category_name}")

        Path(SAVE_DIR_CATEGORY / zip_name).unlink()
        print(f"Removed zip file: {zip_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LaSOT category")

    parser.add_argument(
        "categories",
        nargs="+",
        help="Category Name",
    )

    parser.add_argument(
        "--no-unzip", action="store_true", help="Do not unzip the downloaded files"
    )

    args = parser.parse_args()

    for cat in args.categories:
        download_category(cat, unzip=not args.no_unzip)
