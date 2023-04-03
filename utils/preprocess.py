from setup import downloader, unziper, extractor


def setup_dataset(dataset_url, downloads_path):
    # Download dataset
    print("[1] Downloading dataset...")
    downloader.process([dataset_url], downloads_path)

    # Unzip dataset
    print("[2] Unzipping dataset...")
    unziper.process(downloads_path)

    print("Dataset downloaded and unzipped successfully.")


def extract_images(source_path, dataset_path):
    # Extract images
    print("Extracting images...")
    extractor.process(source_path, dataset_path)

    print("Images cropped successfully.")


def preprocess_image(image):
    return extractor.preprocess_image(image)


def extract_contour(image):
    return extractor.extract_image_contour(image, True)
