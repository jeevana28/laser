import pickle

import os
from dataset_utils.abstract_dataset import AbstractDataset
import json
from PIL import Image

class VQADataset(AbstractDataset):

    def __init__(self, dataset_file="vqa/vqa_val.json", images_dir="coco/"):
        super().__init__()
        self.dataset_file = dataset_file
        self.images_dir = images_dir

    
    def get_dataset(self, logger):
        # Load the VQA dataset questions
        with open(self.dataset_file, "r") as f:
            data = json.load(f)

        dataset = []
        num_dp = len(data)

        count = 0
        for d in data:
            question = d['question']
            answer = d['answer']
            image_path = os.path.join(self.images_dir, d["image"])  # Correct path
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                image.verify()
            else:
                logger.log(f"Image file not found: {image_path} or Invalid image file")
                continue

            count += 1
            if count % 500 == 0:
              print("loading", count, "th image")

            if count == 10:
              break

            dataset.append((image, question, answer))

        logger.log(f"Read VQA dataset of size {num_dp}")


        print("length of dataset in loader = ", len(dataset))
        return dataset