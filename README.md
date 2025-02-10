# Setup

Copy data/valid/captions.csv and data/valid/images into caption_prediction and concept_detection

## Caption Prediction Evaluation

You need to request a licence for UMLS to use caption prediction evaluation. Download UMLS full model (zip file) into caption_prediction/models/MedCAT.

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
│   └── requirements.txt
└── concept_detection
    ├── Dockerfile
    ├── data
    │   └── valid
    │       ├── captions.csv
    │       └── images
    ├── evaluator.py
    └── requirements.txt