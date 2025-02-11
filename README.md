# Setup

Copy data/valid/captions.csv and data/valid/images into caption_prediction and concept_detection

## Caption Prediction Evaluation

You need to request a licence for UMLS to use caption prediction evaluation. Download UMLS full model (zip file) into caption_prediction/models/MedCAT.

```sh
cd caption_prediction
docker build -t caption_prediction_evaluator .
```

Place your submission in caption_prediction, choose device or put all.
TODO delete mount of script only for testing
```sh
docker run --gpus '"device=3"' --rm -v $(pwd)/submission.csv:/app/submission.csv -v $(pwd)/evaluator.py:/app/evaluator.py caption_prediction_evaluator python3 -c "from evaluator import CaptionEvaluator; evaluator = CaptionEvaluator('/app/data/valid/captions.csv'); result = evaluator._evaluate({'submission_file_path': '/app/submission.csv'}); print(result)"
```

## Concept Detection Evaluation

```sh
cd concept_detection
docker build -t concept_detection_evaluator .
```

Place your submission in concept_detection
```sh
docker run --rm -v $(pwd):/app concept_detection_evaluator python -c "from evaluator import ConceptEvaluator; evaluator = ConceptEvaluator('/app/data/valid/concepts.csv'); result = evaluator._evaluate({'submission_file_path': '/app/submission.csv'}); print(result)"
```

## Required File Structure
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
    │       └── concepts.csv
    ├── evaluator.py
    ├── requirements.txt
    └── submission.csv
```