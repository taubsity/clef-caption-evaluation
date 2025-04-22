# Use an official Python runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory inside the container
WORKDIR /app

# Install system dependencies including tini
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    git-lfs \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install MedImageInsights
RUN git lfs install && \
    git clone https://huggingface.co/lion-ai/MedImageInsights && \
    cd MedImageInsights && python3 -m pip install -r requirements.txt && cd ..

RUN chmod -R 755 /app/MedImageInsights

# Install AlignScore
RUN git clone https://github.com/yuh-zha/AlignScore.git && \
    cd AlignScore && python3 -m pip install . && cd .. && \
    python3 -m spacy download en_core_web_sm && \
    mkdir -p models && \
    cd models && \
    git clone https://huggingface.co/yzha/AlignScore && \
    cd ..

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt_tab')"

# Copy the evaluation script
COPY evaluator.py .
COPY medcat_scorer.py .

# Copy model directories (assuming they are available locally)
COPY models/MedCAT models/MedCAT

# Copy data files (ground truth)
COPY data/ data/
