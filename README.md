# Web3Index Backend & AI Solution

## Overview
This repository implements a backend solution for indexing and analyzing Web3 projects. 
It consists of:
- **Web Scraping**: Extracts data from `cryptorank.io`.
- **AI-Enhanced Summaries**: Uses Llama-3 to generate concise project descriptions.
- **Similarity Search**: Identifies the most relevant projects based on a query.
- **Retrieval-Augmented Generation (RAG)**: Provides an AI-enhanced response based on retrieved projects.

## Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Hugging Face token to `config.py`:
   ```python
   HF_TOKEN = "your-huggingface-token"
   ```

## Running the Code
### Part 1: Scraping and AI Summarization
```bash
python part_1.py
```

### Part 2: Querying and AI Response Generation
```bash
python part_2.py
```

## Example Output
The processed project data is saved in `projects.json`. The script finds relevant projects and generates AI-based responses.
