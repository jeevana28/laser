import pickle

import os
from dataset_utils.abstract_dataset import AbstractDataset
import json
from PIL import Image

class NoCapsDataset(AbstractDataset):
    def __init__(self, 
                 dataset_file="nocaps/nocaps_val_4500_captions.json", 
                 images_dir="nocaps/val/", 
                 distribution='out-domain'):
        """
        Initialize the dataset loader.
        Args:
            dataset_file (str): Path to the dataset JSON file.
            images_dir (str): Directory containing the images.
            distribution (str or None): Filter to load only specific distributions.
                                        Options: 'in-domain', 'near-domain', 'out-domain', or None.
        """
        super().__init__()
        self.dataset_file = dataset_file
        self.images_dir = images_dir
        self.distribution = distribution

    def get_dataset(self, logger):
        # Load the dataset JSON file
        with open(self.dataset_file, "r") as f:
            data = json.load(f)

        dataset = []

        # Create a mapping of image IDs to their captions
        captions_dict = {}
        for annotation in data["annotations"]:
            image_id = annotation["image_id"]  # JSON uses "image_id" key for linking
            caption = annotation["caption"]
            if image_id not in captions_dict:
                captions_dict[image_id] = []
            captions_dict[image_id].append(caption)

        # Process each image in the dataset
        count = 0
        for image in data["images"]:
            image_id = image["id"]
            image_file = image["file_name"]
            domain = image["domain"]  # Distribution type (in-domain, near-domain, out-domain)

            # Apply the distribution filter
            if self.distribution and domain != self.distribution:
                continue

            image_path = os.path.join(self.images_dir, image_file)
            if not os.path.exists(image_path):
                logger.log(f"Image file not found: {image_path}")
                continue

            # Ensure the image is valid
            try:
                image_obj = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.log(f"Invalid image file {image_path}: {e}")
                continue

            # Get all captions for this image
            captions = captions_dict.get(image_id, [])
            if not captions:
                logger.log(f"No captions found for Image ID {image_id}.")
                continue

            count += 1
            if count % 500 == 0:
                print(f"Loaded {count} images")

            # Append the image and captions as a tuple
            dataset.append((image_obj, captions))

            # if count==500:
            #     break

        logger.log(f"Loaded {len(dataset)} image-caption pairs from the dataset.")
        print("Total dataset size:", len(dataset))
        return dataset