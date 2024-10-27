import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress
from dataset_utils.coco import CocoDataset
from PIL import Image

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
import json

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

class BLIPExperiment:

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

        val_acc, val_logloss = BLIPExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = BLIPExperiment.get_acc_log_loss(test_predictions)

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
        generated_captions = []
        ground_truth_coco = {
            "images": [],
            "annotations": [],
            "type": "captions",
            "info": {},
            "licenses": []
        }
        annotation_id = 0
        ### resume evaluation code

        # Example loop through your dataset
        for i in tqdm(range(0, dataset_size)):
            
            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            image, answer = dataset[i]  # Assuming dataset[i] returns (image, answer)
            
            # Prepare inputs using the BlipProcessor
            # image = Image.open(image_path).convert
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            question_answer = "Describe the image " + answer
            input_and_answer = processor(images=image, text=question_answer, return_tensors="pt").to(self.device)

            print(input_and_answer['input_ids'].shape)

            with torch.no_grad():
                # Generate from the model
                generate_ids = model_edit.generate(inputs['pixel_values'],
                                                  max_new_tokens=args.max_len,
                                                  min_new_tokens=1)

                generation = processor.decode(generate_ids[0], skip_special_tokens=True)

                # Compute log probability of question + answer
                results = model_edit(input_and_answer['pixel_values'], input_and_answer['input_ids'])
                logits = results.logits[0]  # question + answer length x vocab
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)  # question + answer length x vocab
                print("log prob shape = ", log_prob.shape)
                
                # print(question_answer)
                # question_answer = processor(text=question_answer, return_tensors="pt")
                print("input_and_answer['input_ids'][0] = ", input_and_answer['input_ids'][0].shape)

                print("Answer = ", answer)
                # log_prob_results = self.metrics.answer_log_prob(log_prob=log_prob,
                #                                                 question_answer_token_ids=input_and_answer['input_ids'][0],
                #                                                 answer=answer,
                #                                                 llm_tokenizer=processor)
                # log_prob_results = ContextAnswerLogProb(total_log_prob=log_prob,
                #                                        answer_log_prob=log_prob,
                #                                        answer_len=1)
                log_prob_results = self.metrics.caption_log_prob(log_prob=log_prob,
                                                                question_answer_token_ids=input_and_answer['input_ids'][0],
                                                                answer=answer,
                                                                processor=processor,
                                                                device = self.device)

            # We compute 0-1 match, f1, precision, and recall score in addition to log-prob of the answer tokens
            is_correct = self.metrics.generation_match(generation=generation, answer=answer)
            f1pr_score = self.metrics.f1pr_scores(generation=generation, answer=answer)
            bleu4_score = self.metrics.bleu4(generation=generation, answer=answer)

            ### resume evaluation code
            generated_captions.append({
                'image_id': i,  # Assuming 'i' is unique for each image
                'caption': generation
            })
            # Store the ground-truth caption in COCO-style format
            ground_truth_coco["images"].append({
                "id": i,  # Unique image ID
                "file_name": f"image_{i}.jpg"  # This can be any placeholder, or real image file name
            })
            ground_truth_coco["annotations"].append({
                "image_id": i,
                "id": annotation_id,  # Unique annotation ID
                "caption": answer
            })
            annotation_id += 1 
            ### pause evaluation code

            self.dataset_metric.accept(is_correct=is_correct,
                                      f1pr_score=f1pr_score, bleu4_score=bleu4_score,
                                      log_prob_results=log_prob_results)

            if i % 10 == 0:
                print(f"Image: {image} and gold answer {answer}")
                print(f"{llm_name} generated {generation}")

            predictions_ = {
                "ix": i,
                "image": image,
                "gold-answer": answer,
                "generation": generation,
                "correct": is_correct,
                "f1_score": f1pr_score.f1,
                "precision": f1pr_score.precision,
                "recall": f1pr_score.recall,
                "case-sensitive": self.case_sensitive,  # We ignore case when checking answer
                "white-space-strip": self.strip,        # We ignore white space when checking answer
                "total_logprob": log_prob_results.total_log_prob,
                "answer_logprob": log_prob_results.answer_log_prob,
                "answer_length": log_prob_results.answer_len,
                "question_answer_length": input_and_answer['input_ids'].shape[1]
            }
            predictions.append(predictions_)

        ### resume evaluation code
        output_dir = './evaluation_results'  # Set this to the directory where you want to save results
        os.makedirs(output_dir, exist_ok=True)

        generated_file = os.path.join(output_dir, "generated_captions.json")
        ground_truth_file = os.path.join(output_dir, "ground_truth_captions.json")

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

        # Save results and terminate
        self.terminate_and_save(predictions, coco_eval.eval)
        return predictions

    def terminate_and_save(self, predictions, evaluation):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()
        self.logger.log("debugging")

        time_start = time.time()
        # Save predictions
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"
        save_eval_fname = f"{self.save_dir}/{llm_name}-evaluation-{args.rate}-{args.dtpts}-{args.lnum}.p"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)
        with open(save_eval_fname, "wb") as f:
            pickle.dump(evaluation, f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results["args/%s" % k] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)
        with open(save_summary_fname, "wb") as f:
            json.dump(results, f)

        # Print final numbers and return
        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")


if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with LLAMA 2 LLM on CounterFact')

    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=10, help='maximum length for generation')
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
                        default="/content/drive/MyDrive/cs682/project/coco/results/blip_weights",
                        help="Path where BLIP model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="/content/drive/MyDrive/cs682/project/coco/results/blip_results",
                        help='Directory where the results data is stored')
    parser.add_argument('--dataset_file', type=str,
                        default="/content/drive/MyDrive/cs682/project/coco/",
                        help='Path to the dataset file or directory containing images and captions')

    # below two lines are not present for gptj_bbh
    parser.add_argument('--image_size', type=int, default=256, help='Size to which images will be resized')  # New arg for image processing
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams for beam search during generation')  # Added for better text generation

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "BLIP"
    llm_path = args.model_path
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file

    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lname}-{args.rate}.txt")

    # Step 4: Create an experiment
    experiment = BLIPExperiment(save_dir=save_dir, logger=logger)

    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)

    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    dataset_util = CocoDataset()
    dataset = dataset_util.get_dataset(logger)

    num_dp = len(dataset)
    print("Number of data points = ", num_dp)
    # dataset = []

    # for i in range(num_dp):
    #     question = data[i]["question"]
    #     answer = data[i]["gold-answer"]
    #     assert answer.startswith(" "), f"Found answer that doesn't start with space ${answer}$"
    #     dataset.append((question, answer))
    # logger.log(f"Read dataset of size {num_dp}")

    # Step 6: Run intervention
    base_results = None
    best_results = None
    best_lnum = None
    best_lname = None
    best_rate = None

    # for lnum in [-1, 11]:
    #     print("lnum = ", lnum)
    #     if lnum == -1:
    #         lnames = ["dont"]
    #         rates = [9.9]
    #     else:
    #         lnames = ["fc_in", "fc_out"]
    #         rates = [1.0, 2.0, 9.9]

    #     for lname in lnames:
    #         for rate in reversed(rates):

    #             print("lnum = ", str(lnum), "lname = ", lname, "rates = ", rates)

    #             args.lnum = lnum
    #             args.lname = lname
    #             args.rate = rate
    #             predictions = experiment.intervene(model=model,
    #                                                processor=processor,
    #                                                dataset=dataset,
    #                                                args=args,
    #                                                llm_name=llm_name)

                # results = experiment.validate(predictions, split=0.2)

                # if lname == "dont":
                #     base_results = results
                #     logger.log(f"Base GPTJ => {results.to_str()}")
                # else:
                #     logger.log(f"GPTJ => Layer number: {lnum}, Layer name {lname}, Rate {rate} => "
                #                f"{results.to_str()}")
                #     if best_results is None or \
                #             (results.val_acc > best_results.val_acc) or \
                #             (results.val_acc == best_results.val_acc and results.val_logloss < best_results.val_logloss):

                #         best_results = results
                #         best_lnum = lnum
                #         best_lname = lname
                #         best_rate = rate

                #     logger.log(f"Base model results {base_results.to_str()}. "
                #                f"Best results {best_results.to_str()} at "
                #                f"layer: {best_lnum}, lname: {best_lnum}, rate: {best_rate}")
                #     logger.log("=============")


    args.lnum = 7
    args.lname = "fc_in"
    experiment.intervene(model=model,
                         processor=processor,
                         dataset=dataset,
                         args=args,
                         llm_name=llm_name)

    logger.log("Experimented Completed.")
