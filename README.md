# Document Summarization using PageRank

A text summarization application that uses PageRank algorithm and Universal Sentence Encoder to generate concise summaries of long texts.

## Features

- Text summarization using PageRank algorithm
- Sentence similarity calculation using Universal Sentence Encoder
- Interactive web interface using Gradio
- Support for custom text input and random text generation
- Adjustable summary length

## Requirements

- Python 3.6+
- Dependencies listed in requirements.txt:
  - gradio
  - numpy
  - pandas
  - nltk
  - networkx
  - tensorflow_hub
  - textblob
  - kagglehub
  - matplotlib
  - seaborn
  - transformers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yuva-raja-reddy/document_summarization.git
cd document_summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The web interface will be available at http://localhost:7860

## How it Works

- Text is split into sentences using NLTK
- Universal Sentence Encoder generates embeddings for each sentence
- Sentence similarity matrix is created using cosine similarity
- PageRank algorithm ranks sentences by importance
- Top N sentences are selected for the summary
