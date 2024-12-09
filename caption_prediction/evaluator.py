import logging
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
import base64

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    filename=os.path.join(current_dir, "log.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Redirect stdout and stderr to the logger
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Construct paths to the module directories
aci_bench_evaluation_dir = os.path.join(current_dir, "..", "aci-bench", "evaluation")
med_image_insights_dir = os.path.join(current_dir, "..", "MedImageInsights")

# check if the directories exist
if not os.path.exists(aci_bench_evaluation_dir):
    raise Exception(
        "aci-bench/evaluation directory not found at {}".format(
            aci_bench_evaluation_dir
        )
    )
if not os.path.exists(med_image_insights_dir):
    raise Exception(
        "MedImageInsights directory not found at {}".format(med_image_insights_dir)
    )

# Add these directories to sys.path
sys.path.insert(0, aci_bench_evaluation_dir)
sys.path.insert(0, med_image_insights_dir)

print("Import MEDCON")
from UMLS_evaluation import umls_score_individual

print("Import MedImageInsight")
from medimageinsightmodel import MedImageInsight


class CaptionEvaluator:

    case_sensitive = False

    def __init__(self, ground_truth_path, **kwargs):
        self.ground_truth_path = ground_truth_path
        self.gt = self.load_gt()
        logging.info("Loading ROUGE from HuggingFace")
        self.scorers = {
            "rouge": (evaluate.load("rouge"),),
        }
        idf_sentences = [
            self.preprocess_caption(caption) for caption in self.gt.values()
        ]
        logging.info("Loading BERTScore")
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            idf=True,
            idf_sents=idf_sentences,
            verbose=True,
        )
        logging.info("Loading AlignScore")
        self.align_scorer = AlignScore(
            model="roberta-large",
            batch_size=32,
            device="cuda:0",
            ckpt_path=os.path.join(
                current_dir, "..", "models/AlignScore/AlignScore-base.ckpt"
            ),
            evaluation_mode="nli_sp",
        )
        logging.info("Loading MedImageInsight")
        self.image_similarity_scorer = MedImageInsight(
            model_dir=os.path.join(current_dir, "..", "MedImageInsights/2024.09.27"),
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth",
        )
        self.image_similarity_scorer.load_model()

    def _evaluate(self, client_payload, _context={}):
        logging.info("Evaluating...")
        submission_file_path = client_payload["submission_file_path"]
        predictions = self.load_predictions(submission_file_path)

        alignscore = self.compute_alignscore(predictions)
        bertscore = self.compute_bertscore(predictions)
        rouge = self.compute_rouge(predictions)
        sim = self.compute_similarity(predictions)
        medcon = self.compute_medcon(predictions)

        relevance = np.mean([bertscore, rouge, sim])
        factuality = np.mean([medcon, alignscore])

        _result_object = {
            "score": relevance,
            "score_secondary": factuality,
            "bert": bertscore,
            "rouge": rouge,
            "similarity": sim,
            "medcon": medcon,
            "align": alignscore,
        }

        assert "score" in _result_object
        assert "score_secondary" in _result_object

        return _result_object

    def load_gt(self):
        logging.info("Loading ground truth...")
        pairs = {}
        with open(self.ground_truth_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in tqdm(reader):
                pairs[row[0]] = row[1]
        return pairs

    def load_predictions(self, submission_file_path):
        logging.info("Loading predictions...")
        pairs = {}
        image_ids_gt = set(self.gt.keys())
        occured_images = set()
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile)
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
        logging.info("Computing BERTScore")
        bert_scores = [
            (
                self.bert_scorer.score(
                    predictions=[self.preprocess_caption(candidate_pairs[image_key])],
                    references=[self.preprocess_caption(self.gt[image_key])],
                )[2].item()
                if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0
                else 1
            )
            for image_key in candidate_pairs
        ]
        return np.mean(bert_scores)

    def compute_rouge(self, candidate_pairs):
        logging.info("Computing ROUGE")
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
        logging.info("Computing Alignscore")
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

    def compute_medcon(self, candidate_pairs):
        logging.info("Computing MEDCON")
        medcon_scores = [
            (
                umls_score_individual(self.gt[image_key], candidate_pairs[image_key])
                if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0
                else 1
            )
            for image_key in candidate_pairs
        ]
        return np.mean(medcon_scores)

    def compute_similarity(self, candidate_pairs):
        logging.info("Computing MedImageInsights Similarity")
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


if __name__ == "__main__":
    ground_truth_path = os.path.join(current_dir, "..", "data/valid/captions.csv")
    submission_file_path = os.path.join(current_dir, "..", "data/valid/captions.csv")
    _client_payload = {"submission_file_path": submission_file_path}
    _context = {}
    caption_evaluator = CaptionEvaluator(ground_truth_path)
    result = caption_evaluator._evaluate(_client_payload, _context)
    print(result)
