import csv
import string
import warnings
import numpy as np
import re
import evaluate
from tqdm import tqdm
from alignscore import AlignScore
import sys
import base64
import os
import nltk
import logging
nltk.download('punkt')


# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

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

from UMLS_evaluation import umls_score_individual
from medimageinsightmodel import MedImageInsight

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

# IMAGECLEF 2025 CAPTION - CAPTION PREDICTION
class CaptionEvaluator:

    case_sensitive = False

    def __init__(self, ground_truth_path, **kwargs):
        """
        This is the evaluator class which will be used for the evaluation.
        Please note that the class name should be `CaptionEvaluator`
        `ground_truth` : Holds the path for the ground truth which is used to score the submissions.
        """
        self.ground_truth_path = ground_truth_path
        self.gt = self.load_gt()

        ######## Load Metrics from HuggingFace ########
        print("Loading ROUGE and BERTScore from HuggingFace")
        self.scorers = {
            "rouge": (evaluate.load("rouge"),),
            "bert_scorer": (evaluate.load("bertscore"),),
        }

    def _evaluate(self, client_payload, _context={}):
        # """
        # This is the only method that will be called by the framework
        # returns a _result_object that can contain up to 2 different scores
        # `client_payload["submission_file_path"]` will hold the path of the submission file
        # """
        """
        This method that will be called by the framework returns a _result_object that can contains
        different scores. `client_payload["submission_file_path"]` will hold the path of the submission file.
        """
        print("evaluate...")
        # Load submission file path
        submission_file_path = client_payload["submission_file_path"]
        # Load predictions and validate format
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
        """
        Load and return groundtruth data
        """
        print("loading ground truth...")

        pairs = {}
        with open(self.ground_truth_path) as csvfile:
            reader = csv.reader(csvfile)

            for row in tqdm(reader):
                pairs[row[0]] = row[1]
        return pairs

    def load_predictions(self, submission_file_path):
        """
        Load and return a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
        Validation of the runfile format has to be handled here. simply throw an Exception if there is a validation error.
        """
        print("load predictions...")

        pairs = {}
        image_ids_gt = set(self.gt.keys())
        occured_images = []
        lineCnt = 0
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile)

            for row in tqdm(reader):
                # less than two pipe separated tokens on line => Error
                if len(row) < 2:
                    self.raise_exception(
                        "Wrong format: Each line must consist of an image ID followed by a ',' (comma) and a caption ({}).",
                        lineCnt,
                        "<imageID><comma><caption>",
                    )

                # Image ID does not exist in testset => Error
                image_id = row[0]
                if image_id not in image_ids_gt:
                    self.raise_exception(
                        "Image ID '{}' in submission file does not exist in testset.",
                        lineCnt,
                        image_id,
                    )

                # image id occured at least twice in file => Error
                if image_id in occured_images:
                    self.raise_exception(
                        "Image ID '{}' was specified more than once in submission file.",
                        lineCnt,
                        image_id,
                    )

                pairs[row[0]] = row[1]

                occured_images.append(image_id)
                lineCnt += 1

        # In case not all images from the testset are contained in the file => Error
        if len(occured_images) != len(image_ids_gt):
            self.raise_exception(
                f"Number of image IDs in submission file not equal to number of image IDs in testset.\nNumber in testset: {len(image_ids_gt)}\nNumber in submission: {len(occured_images)}",
                lineCnt,
            )

        return pairs

    def raise_exception(self, message, record_count, *args):
        raise Exception(
            message.format(*args)
            + " Error occured at record line {}.".format(record_count)
        )

    def compute_bertscore(self, candidate_pairs):
        print("BERTScore")
        # Hide warnings
        warnings.filterwarnings("ignore")

        # Remove punctuation from string
        translator = str.maketrans("", "", string.punctuation)

        # Regex for numbers
        number_regex = re.compile(r"\d+")

        bert_scores = []

        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]

            # Optional - Go to lowercase
            if not type(self).case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()

            # replace numbers with the token 'number'
            candidate_caption = number_regex.sub("number", candidate_caption)
            gt_caption = number_regex.sub("number", gt_caption)

            # Remove punctuation using the translator
            candidate_caption = candidate_caption.translate(translator)
            gt_caption = gt_caption.translate(translator)

            # Calculate BERTScore for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_caption) == 0 and len(candidate_caption) == 0:
                    bert_score = 1
                # Calculate the BERTScore
                else:
                    bert_score = self.scorers["bert_scorer"][0].compute(
                        predictions=[candidate_caption],
                        references=[gt_caption],
                        model_type="microsoft/deberta-xlarge-mnli",
                        idf=True,
                    )
            # Handle problematic cases where BERTScore calculation is impossible
            except Exception as e:
                print(e)
                # raise Exception('Problem with {} {}', gt_caption, candidate_caption)
            bert_scores.append(bert_score["recall"])

        return np.mean(bert_scores)

    def compute_rouge(self, candidate_pairs):
        print("ROUGE")
        # Hide warnings
        warnings.filterwarnings("ignore")

        # Remove punctuation from string
        translator = str.maketrans("", "", string.punctuation)

        # Regex for numbers
        number_regex = re.compile(r"\d+")

        rouge_scores = []

        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]

            # Optional - Go to lowercase
            if not type(self).case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()

            # replace numbers with the token 'number'
            candidate_caption = number_regex.sub("number", candidate_caption)
            gt_caption = number_regex.sub("number", gt_caption)

            # Remove punctuation using the translator
            candidate_caption = candidate_caption.translate(translator)
            gt_caption = gt_caption.translate(translator)

            # Calculate ROUGE score for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_caption) == 0 and len(candidate_caption) == 0:
                    rouge1_score_f1 = 1
                # Calculate the ROUGE score
                else:
                    rouge1_score_f1 = self.scorers["rouge"][0].compute(
                        predictions=[candidate_caption],
                        references=[gt_caption],
                        use_aggregator=False,
                        use_stemmer=False,
                    )
            # Handle problematic cases where ROUGE score calculation is impossible
            except Exception as e:
                print(e)
                # raise Exception('Problem with {} {}', gt_caption, candidate_caption)

            # Append the score to the list of scores
            rouge_scores.append(rouge1_score_f1["rouge1"])

        # Calculate the average of all scores
        return np.mean(rouge_scores)

    def compute_alignscore(self, candidate_pairs):
        print("Alignscore")
        # Hide warnings
        warnings.filterwarnings("ignore")

        scorer = AlignScore(
            model="roberta-large",
            batch_size=32,
            device="cuda:0",
            ckpt_path= os.path.join(current_dir,"..","models/AlignScore/AlignScore-base.ckpt"),
            evaluation_mode="nli_sp",
        )
        align_scores = []

        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]

            # Calculate Align score for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_caption) == 0 and len(candidate_caption) == 0:
                    score = 1
                # Calculate the Align score
                else:
                    score = scorer.score(
                        contexts=[gt_caption], claims=[candidate_caption]
                    )

            # Handle problematic cases where ROUGE score calculation is impossible
            except Exception as e:
                print(e)
                # raise Exception('Problem with {} {}', gt_caption, candidate_caption)

            # Append the score to the list of scores
            align_scores.append(score[0])

        return np.mean(align_scores)

    def compute_medcon(self, candidate_pairs):
        print("MEDCON")
        # Hide warnings
        warnings.filterwarnings("ignore")

        medcon_scores = []

        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]

            # Calculate Align score for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_caption) == 0 and len(candidate_caption) == 0:
                    score = 1
                # Calculate the Align score
                else:
                    score = umls_score_individual(gt_caption, candidate_caption)

            # Handle problematic cases where ROUGE score calculation is impossible
            except Exception as e:
                print(e)
                # raise Exception('Problem with {} {}', gt_caption, candidate_caption)

            # Append the score to the list of scores
            medcon_scores.append(score)
        return 0

    def compute_similarity(self, candidate_pairs):
        print("MedImageInsights Similarity")
        # Hide warnings
        warnings.filterwarnings("ignore")

        # Initialize classifier
        classifier = MedImageInsight(
            model_dir= os.path.join(current_dir,"..","MedImageInsights/2024.09.27"),
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth",
        )

        # Load model
        classifier.load_model()

        def read_image(image_path):
            with open(image_path, "rb") as f:
                return f.read()

        image_dir = os.path.join(os.path.dirname(self.ground_truth_path), "images")
        # check if image path exists and throw error if not
        if not os.path.exists(image_dir):
            raise Exception("Image directory does not exist at {}".format(image_dir))

        sim_scores = []

        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]

            image = base64.encodebytes(
                read_image(os.path.join(image_dir, image_key + ".jpg"))
            ).decode("utf-8")
            images = [image]
            texts = [candidate_caption]

            # Image embeddings
            embeddings = classifier.encode(images=images, texts=texts)

            # Text embeddings
            v = embeddings["image_embeddings"][0]
            c = embeddings["text_embeddings"][0]

            # scaling factor
            w = 2.5

            # Compute the cosine similarity between the image vector v and the caption vector c using the formula:
            # cos(c,v)=c⋅v/∥c∥∥v∥​ here ⋅ denotes the dot product and ∥⋅∥ denotes the Euclidean norm.
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_caption) == 0 and len(candidate_caption) == 0:
                    score = 0
                # Calculate the score
                else:
                    cos = np.dot(c, v) / (np.linalg.norm(c) * np.linalg.norm(v))
                    score = w * np.max([cos, 0])

            # Handle problematic cases where score calculation is impossible
            except Exception as e:
                print(e)
                # raise Exception('Problem with {} {}', gt_caption, candidate_caption)

            # Append the score to the list of scores
            sim_scores.append(score)
        return np.mean(sim_scores)


# TEST THIS EVALUATOR
if __name__ == "__main__":
    ground_truth_path = (
         os.path.join(current_dir,"..","data/valid/captions.csv")
    )

    submission_file_path = (
         os.path.join(current_dir,"..","data/valid/captions.csv")
    )

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path

    # Instantiate a dummy context
    _context = {}

    # Instantiate an evaluator
    caption_evaluator = CaptionEvaluator(ground_truth_path)

    # Evaluate
    result = caption_evaluator._evaluate(_client_payload, _context)
    print(result)
