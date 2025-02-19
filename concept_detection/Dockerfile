# Use an official Python runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV DEBIAN_FRONTEND=noninteractive

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

# Copy the evaluation script
COPY evaluator.py .

# Copy data files (ground truth)
COPY data/valid/concepts.csv data/valid/concepts.csv
COPY data/valid/concepts_manual.csv data/valid/concepts_manual.csv