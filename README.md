# Setup

You need docker to run the evaluations with GPU support for caption prediction evaluation.

## Caption Prediction Evaluation

1. Copy `captions.csv` and `images` dir into `caption_prediction/data/valid`.
   
2. Request a licence for UMLS and then download the UMLS full model (zip file) from https://uts.nlm.nih.gov/uts/login?service=https://medcat.rosalind.kcl.ac.uk/auth-callback into `caption_prediction/models/MedCAT`.
   
3. Build the `caption_prediction_evaluator` docker image. 

    ```sh
    cd caption_prediction
    docker build -t caption_prediction_evaluator .
    ```
4. Place your `submission.csv` in `caption_prediction` dir, choose device (GPU) or put all and run the evaluation.
    ```sh
    docker run \
      --gpus '"device=0"' \
      --rm \
      -v $(pwd)/submission.csv:/app/submission.csv \
      caption_prediction_evaluator \
      python3 -c "from evaluator import CaptionEvaluator; evaluator = CaptionEvaluator(); result = evaluator._evaluate({'submission_file_path': '/app/submission.csv'}); print(result)"
    ```

## Concept Detection Evaluation

1. Copy `concepts.csv` into `concept_detection/data/valid`.

2. Build the `concept_detection_evaluator` docker image. 

    ```sh
    cd concept_detection
    docker build -t concept_detection_evaluator .
    ```

3. Place your `submission.csv` in `concept_detection` dir and run evaluation.

    ```sh
    docker run \
      --rm \
      -v $(pwd)/submission.csv:/app/submission.csv \
      concept_detection_evaluator \
      python -c "from evaluator import ConceptEvaluator; evaluator = ConceptEvaluator(); result = evaluator._evaluate({'submission_file_path': '/app/submission.csv'}); print(result)"
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
