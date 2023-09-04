import os
import tarfile


def extract_dataset():
    print("Extracting dataset...")

    if not os.path.exists("Dataset/extracted"):
        os.makedirs("Dataset/extracted", exist_ok=True)

    for split in ["train", "val", "test"]:
        tar = tarfile.open(f"Dataset/{split}.tar.gz", "r:gz")
        tar.extractall(f"Dataset/extracted")
        tar.close()

    print("Dataset extracted successfully!")


extract_dataset()
