import os
import time
import torch
import pickle
import argparse
import re
from word2number import w2n
from num2words import num2words

from tqdm import tqdm
from transformers import BlipProcessor
from transformers import BlipForQuestionAnswering
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress
from dataset_utils.vqa import VQADataset
# from dataset_utils.coco import CocoDataset
from PIL import Image
# from vqa import VQA
# from vqaEval import VQAEval
# from lave_ft5 import LaveFT5

def handle_contractions(text):
    contractions = {
        "dont": "don't",
        "doesnt": "doesn't",
        "didnt": "didn't",
        "cant": "can't",
        "wont": "won't",
        "isnt": "isn't",
        "arent": "aren't",
        "wasnt": "wasn't",
        "werent": "weren't",
        # Add more contractions as needed
    }
    for word, contraction in contractions.items():
        text = re.sub(rf"\b{word}\b", contraction, text)
    return text

# Function to convert number words to digits
def convert_number_words_to_digits(text):
    # Convert words to numbers
    text = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', lambda x: str(w2n.word_to_num(x.group())), text)
    return text

# Function to preprocess text
def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove periods except as decimal points
    text = re.sub(r'(?<=\d)\.(?=\d)', '', text)  # Keep periods in decimals
    text = re.sub(r'\.(?!\d)', '', text)  # Remove other periods

    # 3. Convert number words to digits (for simple words)
    text = convert_number_words_to_digits(text)

    # 4. Remove articles (a, an, the)
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 5. Handle contractions
    text = handle_contractions(text)

    # 6. Replace punctuation except apostrophe and colon with space
    text = re.sub(r"[^\w\s':]", ' ', text)  # Replace all punctuation except ' and :

    # 7. Remove spaces around commas between digits
    text = re.sub(r'(?<=\d),(?=\d)', '', text)  # Remove comma between digits (e.g., 1,000 -> 1000)

    # 8. Ensure only a single space between words
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text

class Results:

    def __init__(self, val_acc, val_logloss, test_acc, test_logloss):
        self.val_acc = val_acc
        self.val_logloss = val_logloss
        self.test_acc = test_acc
        self.test_logloss = test_logloss

    def to_dict(self):
        return {
            "val_acc": self.val_acc,
            "val_logloss": self.val_logloss,
            "test_acc": self.test_acc,
            "test_logloss": self.test_logloss
        }

    def to_str(self, only_test=False):
        if only_test:
            return f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"
        else:
            return f"Validation acc {self.val_acc:.3f}, Validation logloss {self.val_logloss:.3f}, " \
                   f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"

class BLIPVQAExperiment:

    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger

        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)

        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @staticmethod
    def validate(predictions, split=0.2):

        val_size = int(split * len(predictions))
        validation_predictions = predictions[:val_size]
        test_predictions = predictions[val_size:]

        val_acc, val_logloss = BLIPVQAExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = BLIPVQAExperiment.get_acc_log_loss(test_predictions)

        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)
    
    def intervene(self, model, processor, dataset, args, llm_name):

        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

        time_edit_start = time.time()
        model_edit = LaserWrapper.get_edited_model(model=model,
                                                   lname=args.lname,
                                                   lnum=args.lnum,
                                                   rate=args.rate,
                                                   intervention=args.intervention,
                                                   logger=logger,
                                                   in_place=True)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        predictions = []

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        ### resume evaluation code
        # generated_captions = []
        # ground_truth_coco = {
        #     "images": [],
        #     "annotations": [],
        #     "type": "captions",
        #     "info": {},
        #     "licenses": []
        # }
        # annotation_id = 0
        ### resume evaluation code
        correct = 0
        total = 0
        # metric = LaveFT5()

        for i in tqdm(range(0, dataset_size)):
            
            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            image, question, answers = dataset[i]  # Assuming dataset[i] returns (image, answer)
            
            # Prepare inputs using the BlipProcessor
            # image = Image.open(image_path).convert
            inputs = processor(image, question, return_tensors="pt").to(self.device)
            # question_answer = "Describe the image " + answer
            # input_and_answer = processor(images=image, text=question_answer, return_tensors="pt").to(self.device)

            # print(input_and_answer['input_ids'].shape)

            with torch.no_grad():
                # Generate from the model
                generate_ids = model_edit.generate(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'],
                                                  max_new_tokens=args.max_len,
                                                  min_new_tokens=1)

                generation = processor.decode(generate_ids[0], skip_special_tokens=True)
                print(question)
                print(generation)
                generation = preprocess_text(generation)
                # print(generation)

                num_agreeing = answers.count(generation)
                print(num_agreeing)

                # Calculate the accuracy for this question using the formula
                accuracy = min(num_agreeing / 3, 1.0)

                # Accumulate the result
                correct += accuracy
                total += 1
                # lave_score = metric.compute(
                #     prediction=generation,
                #     references=answers,
                #     question=question
                # )
                # print(lave_score)
            if i % 10 == 0:
                print(f"Image: {image} and gold answer {answers}")
                print(f"{llm_name} generated {generation}")

        overall_accuracy = correct / total if total > 0 else 0
        print(overall_accuracy)
        self.terminate_and_save(overall_accuracy)



    def terminate_and_save(self, overall_accuracy):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()
        self.logger.log("debugging")

        time_start = time.time()
        # Save predictions
        # save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"
        save_eval_fname = f"{self.save_dir}/{llm_name}-evaluation-{args.rate}-{args.dtpts}-{args.lnum}.p"

        # with open(save_pred_fname, "wb") as f:
        #     pickle.dump(predictions, f)
        with open(save_eval_fname, "wb") as f:
            pickle.dump(f"Accuracy: {overall_accuracy}", f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results["args/%s" % k] = v


        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        # Print final numbers and return
        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")


if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with BLIP on VQA')

    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=25, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction', 'zero'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out',
                                 'None', 'dont', 'all', 'mlp', 'attn'],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=28, help='Layers to edit', choices=list(range(-1, 33)))


    parser.add_argument('--model_path',
                        type=str,
                        default="vqa/results/blip_weights",
                        help="Path where BLIP model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="vqa/results/blip_results",
                        help='Directory where the results data is stored')
    parser.add_argument('--dataset_file', type=str,
                        default="vqa",
                        help='Path to the dataset file or directory containing images and captions')

    # below two lines are not present for gptj_bbh
    parser.add_argument('--image_size', type=int, default=256, help='Size to which images will be resized')  # New arg for image processing
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams for beam search during generation')  # Added for better text generation

    args = parser.parse_args()

        # Step 2: Load model and tokenizer
    llm_name = "BLIP"
    llm_path = args.model_path
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file

    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lname}-{args.rate}.txt")


    # Step 4: Create an experiment
    experiment = BLIPVQAExperiment(save_dir=save_dir, logger=logger)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    dataset_util = VQADataset()
    dataset = dataset_util.get_dataset(logger)

    num_dp = len(dataset)
    print("Number of data points = ", num_dp)
    experiment.intervene(model=model,
                         processor=processor,
                         dataset=dataset,
                         args=args,
                         llm_name=llm_name)

    logger.log("Experimented Completed.")