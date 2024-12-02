import pickle

import os
from dataset_utils.abstract_dataset import AbstractDataset
import json
from PIL import Image

class CocoDataset(AbstractDataset):

    def __init__(self, dataset_file="coco/annotations/coco_karpathy_val_final.json", images_dir="coco/"):
        super().__init__()
        self.dataset_file = dataset_file
        self.images_dir = images_dir

    def get_dataset(self, logger):
        # Load the COCO dataset annotations
        with open(self.dataset_file, "r") as f:
            data = json.load(f)

        dataset = []
        num_dp = len(data)

        count = 0
        for annotation in data:
            # print(annotation)
            image_id = annotation["image"]
            captions = annotation["captions"]
            # assert caption.startswith(" "), f"Found caption that doesn't start with space ${caption}$"

            # Find the corresponding image file
            # image_file = next((img["file_name"] for img in data["images"] if img["id"] == image_id), None)
            # if image_file is None:
            #     logger.log(f"Image ID {image_id} not found.")
            #     continue

            image_path = os.path.join(self.images_dir, image_id)
            if not os.path.exists(image_path):
                logger.log(f"Image file not found: {image_path}")
                continue

            # Open image to ensure it's valid (optional)
            try:
                image = Image.open(image_path).convert("RGB")
                image.verify()  # Verify if it's a valid image
            except Exception as e:
                logger.log(f"Invalid image file {image_path}: {e}")
                continue
            
            count += 1
            if count % 500 == 0:
              print("loading", count, "th image")

            if count == 2500:
              break

            dataset.append((image, captions))

        logger.log(f"Read COCO dataset of size {num_dp}")


        print("length of dataset in loader = ", len(dataset))
        return dataset