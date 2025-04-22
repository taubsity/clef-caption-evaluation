import os
import sys
import csv
import string
import numpy as np
import re
import evaluate
from tqdm import tqdm
from alignscore import AlignScore
from bert_score import BERTScorer
from medcat_scorer import MedCatScorer
import base64
import torch
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the module directories
med_image_insights_dir = os.path.join(current_dir, "MedImageInsights")

# check if the directory exist
if not os.path.exists(med_image_insights_dir):
    raise Exception(
        "MedImageInsights directory not found at {}".format(med_image_insights_dir)
    )

sys.path.insert(0, med_image_insights_dir)

from medimageinsightmodel import MedImageInsight


class CaptionEvaluator:

    case_sensitive = False

    def __init__(self, ground_truth_path="/app/data/valid/captions.csv", **kwargs):
        print("Initializing evaluator...")
        self.ground_truth_path = ground_truth_path
        self.gt = self.load_gt()
        print("Loading ROUGE from HuggingFace")
        self.scorers = {
            "rouge": (evaluate.load("rouge"),),
        }
        idf_sentences = [
            self.preprocess_caption(caption) for caption in self.gt.values()
        ]
        print("Loading BERTScore")
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            idf=True,
            idf_sents=idf_sentences,
        )
        print("Loading MedCatScorer")
        self.medcat_scorer = MedCatScorer(
            model_path=os.path.join(
                current_dir,
                [
                    os.path.join("models/MedCAT", filename)
                    for filename in os.listdir("models/MedCAT")
                    if filename.endswith(".zip")
                ][0],
            )
        )
        print("Loading AlignScore")
        self.align_scorer = AlignScore(
            model="roberta-large",
            batch_size=32,
            device="cuda:0",
            ckpt_path=os.path.join(
                current_dir, "models/AlignScore/AlignScore-base.ckpt"
            ),
            evaluation_mode="nli_sp",
            verbose=False,
        )
        print("Loading MedImageInsight")
        self.image_similarity_scorer = MedImageInsight(
            model_dir=os.path.join(current_dir, "MedImageInsights/2024.09.27"),
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth",
        )
        print("Loading BLEURT")
        self.bleurt_config = BleurtConfig.from_pretrained("lucadiliello/BLEURT-20-D12")
        self.bleurt_model = BleurtForSequenceClassification.from_pretrained(
            "lucadiliello/BLEURT-20-D12"
        )
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(
            "lucadiliello/BLEURT-20-D12"
        )

    def _evaluate(self, client_payload, _context={}):
        print("Evaluating...")
        submission_file_path = client_payload["submission_file_path"]
        predictions = self.load_predictions(submission_file_path)

        print("Compute BERTScore")
        bertscore = self.compute_bertscore(predictions)
        print("BERTScore:", bertscore)
        print("Compute AlignScore")
        alignscore = self.compute_alignscore(predictions)
        print("AlignScore:", alignscore)
        print("Compute ROUGE")
        rouge = self.compute_rouge(predictions)
        print("ROUGE:", rouge)
        print("Compute Image-Caption Similarity")
        sim = self.compute_similarity(predictions)
        print("Similarity:", sim)
        print("Compute BLEURT")
        bleurt = self.compute_bleurt(predictions)
        print("BLEURT:", bleurt)
        print("Compute MedCAT")
        medcats = self.compute_medcats(predictions)
        print("Medcats:", medcats)

        relevance = np.mean([bertscore, rouge, sim])
        factuality = np.mean([medcats, alignscore])

        _result_object = {
            "score": relevance,
            "score_secondary": factuality,
            "bert": bertscore,
            "rouge": rouge,
            "similarity": sim,
            "bleurt": bleurt,
            "medcat": medcats,
            "align": alignscore,
        }
        print(
            "Similarity,BERTScore,ROUGE,BLEURT,Relevance,Medcats,AlignScore,Factuality\n"
            + "{},{},{},{},{},{},{},{}\n".format(
                sim,
                bertscore,
                rouge,
                bleurt,
                relevance,
                medcats,
                alignscore,
                factuality,
            )
        )

        assert "score" in _result_object
        assert "score_secondary" in _result_object

        return _result_object

    def load_gt(self):
        print("Loading ground truth...")
        pairs = {}
        with open(self.ground_truth_path) as csvfile:
            reader = csv.reader(csvfile)
            first_line = next(reader)
            # Check if it's a header
            if first_line and first_line[0].lower() != "id":
                # Process the first line if it's not a header
                pairs[first_line[0]] = first_line[1]
            for row in tqdm(reader):
                pairs[row[0]] = row[1]
        return pairs

    def load_predictions(self, submission_file_path):
        print("Loading predictions...")
        pairs = {}
        image_ids_gt = set(self.gt.keys())
        occured_images = set()
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile)
            first_line = next(reader)
            # Check if it's a header
            if first_line and first_line[0].lower() != "id":
                # Process the first line if it's not a header
                pairs[first_line[0]] = first_line[1]
            for lineCnt, row in enumerate(tqdm(reader)):
                if len(row) < 2:
                    self.raise_exception(
                        "Wrong format: Each line must consist of an image ID followed by a ',' (comma) and a caption ({}).",
                        lineCnt,
                        "<imageID><comma><caption>",
                    )
                image_id = row[0]
                if image_id not in image_ids_gt:
                    self.raise_exception(
                        "Image ID '{}' in submission file does not exist in testset.",
                        lineCnt,
                        image_id,
                    )
                if image_id in occured_images:
                    self.raise_exception(
                        "Image ID '{}' was specified more than once in submission file.",
                        lineCnt,
                        image_id,
                    )
                pairs[image_id] = row[1]
                occured_images.add(image_id)
        if len(occured_images) != len(image_ids_gt):
            self.raise_exception(
                f"Number of image IDs in submission file not equal to number of image IDs in testset.\nNumber in testset: {len(image_ids_gt)}\nNumber in submission: {len(occured_images)}",
                lineCnt,
            )
        return pairs

    def raise_exception(self, message, record_count, *args):
        raise Exception(
            message.format(*args)
            + " Error occurred at record line {}.".format(record_count)
        )

    def preprocess_caption(self, caption):
        translator = str.maketrans("", "", string.punctuation)
        number_regex = re.compile(r"\d+")
        if not type(self).case_sensitive:
            caption = caption.lower()
        caption = number_regex.sub("number", caption)
        caption = caption.translate(translator)
        return caption

    def compute_bertscore(self, candidate_pairs):
        print("Computing BERTScore")
        bert_scores = [
            (
                self.bert_scorer.score(
                    cands=[self.preprocess_caption(candidate_pairs[image_key])],
                    refs=[self.preprocess_caption(self.gt[image_key])],
                )[2].item()
                if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0
                else 1
            )
            for image_key in candidate_pairs
        ]
        return np.mean(bert_scores)

    def compute_rouge(self, candidate_pairs):
        print("Computing ROUGE")
        rouge_scores = [
            (
                self.scorers["rouge"][0].compute(
                    predictions=[self.preprocess_caption(candidate_pairs[image_key])],
                    references=[self.preprocess_caption(self.gt[image_key])],
                    use_aggregator=False,
                    use_stemmer=False,
                )["rouge1"]
                if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0
                else 1
            )
            for image_key in candidate_pairs
        ]
        return np.mean(rouge_scores)

    def compute_alignscore(self, candidate_pairs):
        print("Computing Alignscore")
        align_scores = [
            (
                self.align_scorer.score(
                    contexts=[self.gt[image_key]], claims=[candidate_pairs[image_key]]
                )[0]
                if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0
                else 1
            )
            for image_key in candidate_pairs
        ]
        return np.mean(align_scores)

    def compute_medcats(self, candidate_pairs):
        print("Computing MEDCATS")
        medcat_scores = []
        for image_key in candidate_pairs:
            if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0:
                score = self.medcat_scorer.score(
                    self.gt[image_key], candidate_pairs[image_key]
                )
            else:
                score = 1
            medcat_scores.append(score)
        return np.mean(medcat_scores)

    def compute_similarity(self, candidate_pairs):
        print("Computing MedImageInsights Similarity")
        self.image_similarity_scorer.load_model()

        image_dir = os.path.join(os.path.dirname(self.ground_truth_path), "images")
        if not os.path.exists(image_dir):
            raise Exception("Image directory does not exist at {}".format(image_dir))
        images = {
            image_key: base64.encodebytes(
                open(os.path.join(image_dir, image_key + ".jpg"), "rb").read()
            ).decode("utf-8")
            for image_key in candidate_pairs
        }
        sim_scores = []
        for image_key in candidate_pairs:
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]
            try:
                if len(gt_caption) == 0 and len(candidate_caption) == 0:
                    score = 1
                else:
                    embeddings = self.image_similarity_scorer.encode(
                        images=[images[image_key]], texts=[candidate_caption]
                    )
                    v = embeddings["image_embeddings"][0]
                    c = embeddings["text_embeddings"][0]
                    w = 2.5
                    cos = np.dot(c, v) / (np.linalg.norm(c) * np.linalg.norm(v))
                    score = w * np.max([cos, 0])
            except Exception as e:
                logging.error(e)
                score = 1
            sim_scores.append(score)
        return np.mean(sim_scores)

    def compute_bleurt(self, candidate_pairs):
        print("Computing BLEURT")
        references = [
            self.preprocess_caption(self.gt[image_key]) for image_key in candidate_pairs
        ]
        candidates = [
            self.preprocess_caption(candidate_pairs[image_key])
            for image_key in candidate_pairs
        ]
        self.bleurt_model.eval()
        with torch.no_grad():
            inputs = self.bleurt_tokenizer(
                references,
                candidates,
                padding="longest",
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            res = self.bleurt_model(**inputs).logits.flatten().tolist()
        return np.mean(res)


if __name__ == "__main__":
    print("Testing evaluator...")
    ground_truth_path = os.path.join(current_dir, "data/valid/captions.csv")
    submission_file_path = os.path.join(
        current_dir, "data/valid/captions.csv"
    )  # change this to the path of the submission file
    _client_payload = {"submission_file_path": submission_file_path}
    _context = {}
    caption_evaluator = CaptionEvaluator(ground_truth_path)
    result = caption_evaluator._evaluate(_client_payload, _context)
    print(result)
