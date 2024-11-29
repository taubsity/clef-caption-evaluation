import csv
import string
import warnings
import numpy as np
import re
import evaluate

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
    print('Loading ROUGE and BERTScore from HuggingFace')
    self.scorers = {
        'rouge': (
            evaluate.load('rouge'),
        ),
        'bert_scorer': (
            evaluate.load('bertscore'),
        )}


def _evaluate(self, client_payload, _context={}):
    """
    This is the only method that will be called by the framework
    returns a _result_object that can contain up to 2 different scores
    `client_payload["submission_file_path"]` will hold the path of the submission file
    """
    print("evaluate...")
    # Load submission file path
    submission_file_path = client_payload["submission_file_path"]
    # Load preditctions and validate format
    predictions = self.load_predictions(submission_file_path)

    bleu_score = self.compute_primary_score(predictions)
    score_secondary = self.compute_secondary_score(predictions)

    _result_object = {
        "score": bleu_score,
        "score_secondary": score_secondary
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

        # Check if the first line is a header and skip it
        first_line = next(reader)

        if "ID" in first_line[0]:
            pass
        else:
            pairs[first_line[0]] = first_line[1]

        for row in reader:
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
    with open(submission_file_path) as csvfile:
        reader = csv.reader(csvfile)

        lineCnt = 0
        occured_images = []
        for row in reader:
            if "ID" not in row[0]:
                lineCnt += 1

                # less than two pipe separated tokens on line => Error
                if(len(row) < 2):
                    self.raise_exception("Wrong format: Each line must consist of an image ID followed by a ',' (comma) and a caption ({}).",
                                            lineCnt, "<imageID><comma><caption>")

                    image_id = row[0]

                    # Image ID does not exist in testset => Error
                    if image_id not in image_ids_gt:
                        self.raise_exception(
                            "Image ID '{}' in submission file does not exist in testset.", lineCnt, image_id)
                        # raise Exception("Image ID '{}' in submission file does not exist in testset {}"
                        #     .format(image_id,self.line_nbr_string(lineCnt)))

                    # image id occured at least twice in file => Error
                    if image_id in occured_images:
                        self.raise_exception(
                            "Image ID '{}' was specified more than once in submission file.", lineCnt, image_id)
                        # raise Exception("Image ID '{}' was specified more than once in submission file {}"
                        #     .format(image_id, self.line_nbr_string(lineCnt)))

                    occured_images.append(image_id)

                    pairs[row[0]] = row[1]

            # In case not all images from the testset are contained in the file => Error
            if(len(occured_images) != len(image_ids_gt)):
                self.raise_exception(
                    "Number of image IDs in submission file not equal to number of image IDs in testset.", lineCnt)

        return pairs

    def raise_exception(self, message, record_count, *args):
        raise Exception(message.format(
            *args)+" Error occured at record line {}.".format(record_count))

    def compute_primary_score(self, predictions):
        """
        Compute and return the primary score
        `predictions` : valid predictions in correct format
        NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
        Valiation should be handled in the load_predictions method
        """
        print("compute primary score...")

        return self.compute_bertscore(predictions)

    def compute_bertscore(self, candidate_pairs):
        # Hide warnings
        warnings.filterwarnings('ignore')

        # Remove punctuation from string
        translator = str.maketrans('', '', string.punctuation)

        # Regex for numbers
        number_regex = re.compile(r'\d+')

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
            candidate_caption = number_regex.sub('number', candidate_caption)
            gt_caption = number_regex.sub('number', gt_caption)

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
                    bert_score = self.scorers["bert_scorer"][0].compute(predictions=[candidate_caption], references=[gt_caption], model_type='microsoft/deberta-xlarge-mnli')
            # Handle problematic cases where BERTScore calculation is impossible
            except Exception as e:
                print(e)
                #raise Exception('Problem with {} {}', gt_caption, candidate_caption)
            bert_scores.append(bert_score["f1"])

        return np.mean(bert_scores)

    def compute_secondary_score(self, predictions):
        """
        Compute and return the secondary score
        Ignore or remove this method if you do not have a secondary score to provide
        `predictions` : valid predictions in correct format
        NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
        Valiation should be handled in the load_predictions method
        """
        print("compute secondary score...")

        return self.compute_rouge(predictions)

        # PUT AUXILIARY METHODS BELOW
        # ...
        # ...
    def compute_rouge(self, candidate_pairs):
        # Hide warnings
        warnings.filterwarnings('ignore')

        # Remove punctuation from string
        translator = str.maketrans('', '', string.punctuation)

        # Regex for numbers
        number_regex = re.compile(r'\d+')

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
            candidate_caption = number_regex.sub('number', candidate_caption)
            gt_caption = number_regex.sub('number', gt_caption)

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
                    rouge1_score_f1 = self.scorers["rouge"][0].compute(predictions=[candidate_caption],
                                                                       references=[gt_caption], use_aggregator=False, use_stemmer=False)
            # Handle problematic cases where ROUGE score calculation is impossible
            except Exception as e:
                print(e)
                #raise Exception('Problem with {} {}', gt_caption, candidate_caption)

            # Append the score to the list of scores
            rouge_scores.append(rouge1_score_f1["rouge1"])

        # Calculate the average of all scores
        return np.mean(rouge_scores)


# TEST THIS EVALUATOR
if __name__ == "__main__":

    ground_truth_path = "caption_prediction_test_gt.csv"

    submission_file_path = "submission.csv"

    _client_payload = {}
    _client_payload["submission_file_path"] = submission_file_path

    # Instaiate a dummy context
    _context = {}

    # Instantiate an evaluator
    caption_evaluator = CaptionEvaluator(ground_truth_path)

    # Evaluate
    result = caption_evaluator._evaluate(_client_payload, _context)
    print(result)
