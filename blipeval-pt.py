import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from PIL import Image

# Load the Karpathy COCO split JSON
karpathy_split_path = "/Users/jeevana/Documents/GitHub/laser/coco/annotations/captions_val2014_sampled_karpathy.json"  # Update this path
with open(karpathy_split_path, "r") as f:
    karpathy_data = json.load(f)

# Extract the validation set (karpathy_data is a list of dictionaries)
val_data = karpathy_data

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate captions
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Prepare ground truth captions (from COCO Karpathy validation set)
generated_captions = []
ground_truth_coco = []
annotation_id = 0

# Process the first 5 images in the validation set
for i, example in enumerate(val_data["annotations"][:10]):
    image_path = example["image"]  # This path should be correct based on your system
    image = Image.open("/Users/jeevana/Documents/GitHub/laser/coco/" + image_path).convert("RGB")
    image_id = int(example["image"].split('_')[-1].split('.')[0])  # Extract image ID as integer
    caption = generate_caption(image)  # Generate caption for the image
    gt_caption = example["caption"]

    # Append the generated caption in the correct format
    generated_captions.append({
        'image_id': image_id,  # Using image_id directly instead of index
        'caption': caption
    })

    print("generated caption = ", caption)
    print("gt caption = ", gt_caption)

    # Store the ground-truth captions in COCO-style format
    ground_truth_coco.append({
        "id": image_id,  # Unique image ID
        "caption": gt_caption  # This can be any placeholder, or real image file name
    })

# Write JSON files
generated_file = "generated_captions.json"
ground_truth_file = "ground_truth_captions.json"
with open(generated_file, 'w') as f:
    json.dump(generated_captions, f)
with open(ground_truth_file, 'w') as f:
    json.dump(ground_truth_coco, f)

# Evaluate using pycocoevalcap
coco = COCO(ground_truth_file)
coco_result = coco.loadRes(generated_file)
coco_eval = COCOEvalCap(coco, coco_result)
coco_eval.evaluate()

for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.3f}')
