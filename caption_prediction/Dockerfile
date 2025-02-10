# Use an official Python runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV DEBIAN_FRONTEND=noninteractive
ENV GROUND_TRUTH_PATH="/app/data/valid/captions.csv"

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install BLEURT
RUN git clone https://github.com/google-research/bleurt.git && \
    cd bleurt && pip install . && cd ..

# Install AlignScore
RUN git clone https://github.com/yuh-zha/AlignScore.git && \
    cd AlignScore && pip install . && cd .. && \
    python -m spacy download en_core_web_sm && \
    mkdir -p models && \
    cd models && \
    git clone https://huggingface.co/yzha/AlignScore && \
    cd ..

# Install MedImageInsights
RUN git lfs install && \
    git clone https://huggingface.co/lion-ai/MedImageInsights && \
    cd MedImageInsights && pip install -r requirements.txt && cd ..

# Copy the evaluation script
COPY caption_prediction/evaluator.py .
COPY caption_prediction/medcat_scorer.py .

# Copy model directories (assuming they are available locally)
COPY models/MedCAT models/MedCAT
COPY models/AlignScore models/AlignScore
COPY MedImageInsights MedImageInsights

# Copy data files (ground truth)
COPY data/valid/captions.csv data/valid/captions.csv
COPY data/valid/images/ data/valid/images/

# Run the container with a test submission file to verify everything loads correctly
CMD ["python", "-c", "from evaluator import CaptionEvaluator; evaluator = CaptionEvaluator(); print('Evaluator initialized')"]
