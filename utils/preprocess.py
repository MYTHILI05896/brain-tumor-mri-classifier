import os

import cv2
import numpy as np
from tqdm import tqdm

from setup import downloader, unziper, extractor


def setup_dataset(dataset_url, downloads_path):
    # Download dataset
    print("[1] Downloading dataset...")
    downloader.process([dataset_url], downloads_path)

    # Unzip dataset
    print("[2] Unzipping dataset...")
    unziper.process(downloads_path)

    print("Dataset downloaded and unzipped successfully.")


def extract_contour(image):
    return extractor.extract_contour(image, True)


def extract_images(source_path, dataset_path):
    # Extract images
    print("Extracting images...")
    extractor.process(source_path, dataset_path)

    print("Images cropped successfully.")


def preprocess_data(source_path, destination_path):
    extractor.clear_screen()
    extractor.make_folder(destination_path)

    for root, dirs, files in os.walk(source_path):
        sub_dir = os.path.relpath(root, source_path)
        extractor.make_folder(os.path.join(destination_path, sub_dir))

        # extract each file and update the progress bar
        if files:
            progress_bar = tqdm(total=len(files), desc=f'Processing {sub_dir} images')
            for file in files:
                if not file.endswith('.jpg'):
                    continue

                # Read the original image
                original_path = os.path.join(root, file)
                img = cv2.imread(original_path)
                processed_img = preprocess_image(img)

                # Save the optimized image
                processed_path = os.path.join(destination_path, sub_dir, file)
                cv2.imwrite(processed_path, processed_img)
                progress_bar.update(1)

            progress_bar.close()


def preprocess_image(image):
    # Resize image to 256x256
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    # Normalize the image
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # remove images noise.
    image = cv2.bilateralFilter(image, 2, 50, 50)

    return image

# def save_image(image, path):
