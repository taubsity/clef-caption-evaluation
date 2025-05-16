# Setup

You need docker to run the evaluations with GPU support for caption prediction evaluation. Please refer to the last section of this README to see submission format instructions.

## Caption Prediction Evaluation

1. Copy `captions.csv` and `images` dir into `caption_prediction/data/valid`.
   
2. Request a licence for UMLS and then download the UMLS full model (zip file) from https://uts.nlm.nih.gov/uts/login?service=https://medcat.rosalind.kcl.ac.uk/auth-callback into `caption_prediction/models/MedCAT`.
   
3. Build the `caption_prediction_evaluator` docker image. 

    ```sh
    cd caption_prediction
    docker build --no-cache -t caption_prediction_evaluator .
    ```
4. Place your `submission.csv` in `caption_prediction` dir, choose device (GPU) or put all and run the evaluation.
    ```sh
    docker run \
      --gpus '"device=1"' \
      --rm \
      -v $(pwd)/submission.csv:/app/submission.csv \
      caption_prediction_evaluator \
      python3 -c "from evaluator import CaptionEvaluator; evaluator = CaptionEvaluator(); result = evaluator._evaluate({'submission_file_path': '/app/submission.csv'}); print(result)"
    ```
   Submission format: `submission.csv` with the two columns **ID** and **Caption**.

   (`submission.csv` is the file you submit to AI4MediaBench in a .zip archive)

   Example:
    ```plain
    ID,Caption
    ImageCLEFmedical_Caption_2025_valid_0,"Illustration of the original image with an ROI.T: Tumor."
    ```

## Concept Detection Evaluation

1. Copy `concepts.csv` and `concepts_manual.csv` into `concept_detection/data/valid`.

2. Build the `concept_detection_evaluator` docker image. 

    ```sh
    cd concept_detection
    docker build -t concept_detection_evaluator .
    ```

3. Place your `submission.csv` in `concept_detection` dir and run evaluation.

    ```sh
    docker run \
      --rm \
      -v $(pwd)/johanna.csv:/app/submission.csv \
      concept_detection_evaluator \
      python -c "from evaluator import ConceptEvaluator; evaluator = ConceptEvaluator(); result = evaluator._evaluate({'submission_file_path': '/app/submission.csv'}); print(result)"
    ```
    Submission format: `submission.csv` with the two columns **ID** and **CUIs** with semicolon seperated CUIs. 

   (`submission.csv` is the file you submit to AI4MediaBench in a .zip archive)

   Example:
   ```plain
   ID,CUIs
   ImageCLEFmedical_Caption_2024_valid_000001,C0040405;C0856747
   ```

## File Structure

This is how the file structure would look like with UMLS model and submission.csv files:

```plain
.
├── README.md
├── caption_prediction
│   ├── Dockerfile
│   ├── data
│   │   └── valid
│   │       ├── captions.csv
│   │       └── images
│   ├── evaluator.py
│   ├── medcat_scorer.py
│   ├── models
│   │   └── MedCAT
│   │       └── umls_self_train_model_pt2ch_3760d588371755d0.zip
│   ├── requirements.txt
|   └── submission.csv
└── concept_detection
    ├── Dockerfile
    ├── data
    │   └── valid
    │       ├── concepts.csv
    │       └── concepts_manual.csv
    ├── evaluator.py
    ├── requirements.txt
    └── submission.csv
```

# Submission Format Instructions

Please note the following when using the scripts or submitting to the ai4mediabench website.

* ai4mediabench requires a zip file (name it as you like) with only your submission in csv format (named "submission.csv").
* Your submission can only contain two columns ID,Caption or ID,CUIs.
* The submission check on the site is very strict and does not allow extra (even empty) lines in the file.
* Image IDs can only appear once in the file.
* Only include IDs from the official test set.
* Ensure standard utf-8 encoding.

* For caption prediction:
  * Use quotation marks for all captions, or at least for those that contain a comma (,).

* For concept detection:
  * Remove duplicate CUIs on a line. The same concept can not be specified more than once for the same image ID.
  * Use ; to separate CUIs.