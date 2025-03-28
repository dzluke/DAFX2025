# Unsupervised Text-to-Sound Mapping via Embedding Space Alignment


Luke Dzwonczyk and Carmine-Emanuele Cella

CNMAT, UC Berkeley


## Abstract
This work focuses on developing an artistic tool that performs an unsupervised mapping between text and sound, converting an input text string into a series of sounds from a given sound corpus. With the use of a pre-trained sound embedding model and a separate, pre-trained text embedding model, the challenge is to find a mapping between the two feature spaces. Our approach is unsupervised, which removes the need for labeled sound data, avoiding potential issues in labeling musical data and allowing any sound corpus to be used with the system. The tool performs the task of text-to-sound retrieval, creating a soundfile in which each word in the text input is mapped to a single sound in the corpus, and the resulting sounds are concatenated to play sequentially. We experiment with three different mapping methods, and perform quantitative and qualitative evaluations on the outputs. Our results demonstrate the potential of unsupervised methods for creative applications in text-to-sound mapping.

---

## Installation Instructions

### Prerequisites
- Python 3.10 or higher
- Conda package manager
- CUDA-enabled GPU (optional but recommended for faster processing)

### Setting Up the Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/dzluke/DAFX2025
   cd DAFX2025
   ```

2. Create a Conda environment:
   ```bash
   conda create -n dafx25_env python=3.10 -y
   conda activate dafx25_env
   ```

3. Install the required dependencies:
   ```bash
     # TODO
   ```

4. Install additional models:
   - Download the `fastText` model:
     ```bash
     # for Windows
     mkdir -p fastText
     wget -P fastText https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
     gunzip fastText/cc.en.300.bin.gz
     ```
   - Install `word2vec` and `MuQ`. `RoBERTa` is included when you install `transformers`

---

## Usage Guide

### Running the Code
1. Prepare your input data:
   - Place your sound corpus in a directory (e.g., `./corpora/sound/choir`).
   - Provide a text corpus as a `.txt` file (e.g., `./corpora/text/haiku.txt`).

2. In `main.py` set your desired parameters and run it to generate sound outputs:
   ```bash
   python main.py
   ```
   This will process the text and sound corpora, map the embeddings, and save the generated sound output in the `output/` directory.

3. To perform batch evaluations across multiple corpora and parameter combinations, use:
   ```bash
   python batch.py
   ```

4. To analyze and visualize evaluation results, use:
   ```bash
   python eval.py
   ```

---

## Repository Overview

### Purpose of Each File
- **`main.py`**: The primary script for running the text-to-sound mapping pipeline. It handles preprocessing, embedding, mapping, and evaluation.
- **`models.py`**: Contains implementations of various embedding models for text and sound, including `MuQ`, `RoBERTa`, `fastText`, and `CLAP`.
- **`util.py`**: Provides utility functions for parameter management, sound preprocessing, and file handling.
- **`batch.py`**: Automates the evaluation of multiple parameter combinations across different corpora.
- **`eval.py`**: Implements the evaluation framework for analyzing and visualizing the results of the mapping process.

---

## Detailed Explanation of Python Files

### `main.py`
This file orchestrates the entire text-to-sound mapping pipeline. Key functions include:
- **`embed_sounds(sounds, encoder, cache_file=None)`**: Embeds sound samples using the specified encoder and caches the results.
- **`embed_text(text, encoder, cache_file=None)`**: Embeds text data using the specified encoder and caches the results.
- **`run(params, evaluator=None, cache=True, save_sound=False, normalize=False)`**: Executes the mapping process for a given set of parameters, evaluates the results, and optionally saves the output sound.

### `models.py`
This file defines the embedding models used for text and sound.

### `util.py`
This file provides utility functions for preprocessing and parameter management. Key components include:
- **`Parameter`**: A class representing a single set of parameters for a run.
- **`ParameterGenerator`**: Generates all possible parameter combinations for evaluation.

### `batch.py`
This file automates the evaluation of multiple parameter combinations. Key functions include:
- **`run_comprehensive_evaluation(check_cache=True)`**: Runs evaluations across multiple corpora and parameter combinations, saving the results to a file.

### `eval.py`
This file implements the evaluation framework for analyzing results. Key components include:
- **`EvaluationFramework`**: A class for managing evaluation results, generating reports, and visualizing data.
  - **`save_result(params, scores)`**: Saves evaluation results to .csv for a specific parameter set.
- **`analyze_experiments(results_file, filter_params, compare_param)`**: Analyzes the results and provides insights into parameter impacts.
- **`generate_plots(results_file, output_dir, filter_params, compare_param)`**: Generates visualizations for evaluation metrics.

---
