# Text Analysis of Entrepreneurial Culture & Isomorphism in Korean Universities

## Overview
This research project analyzes the relationship between entrepreneurial culture, institutional isomorphism, and organizational performance in Korean universities using machine learning-based text analysis. The study focuses on examining university reports using natural language processing and topic modeling techniques.

## Research Process
1. Text Preprocessing
   - Special character removal and text cleaning
   - Morphological analysis using Mecab
   - Custom dictionary enhancement for academic terms

2. Topic Analysis
   - Co-occurrence analysis of key terms
   - Advanced topic modeling using KoBERT+CTM
   - Similarity analysis using KR-SBERT

## Directory Structure
```
├── data/               
│   ├── raw/           # Original university reports
│   └── processed/     # Preprocessed text data
├── notebooks/         
│   ├── preprocessing.ipynb  # Text preprocessing steps
│   └── analysis.ipynb      # Topic modeling and analysis
└── src/               
    ├── preprocess.py  # Text preprocessing utilities
    └── modeling.py    # Topic modeling implementation
```

## Requirements
```python
numpy>=1.19.2
pandas>=1.2.0
konlpy>=0.6.0
torch>=1.9.0
transformers>=4.11.0
contextualized-topic-models>=2.2.0
sentence-transformers>=2.2.0
```

## Installation

1. Install MeCab and related dependencies:
```bash
apt-get update
apt-get install -y git make curl xz-utils file
apt-get install -y mecab libmecab-dev mecab-ipadic-utf8 python3-dev
```

2. Install Python packages:
```bash
pip install -r requirements.txt
```

3. Install Korean language resources:
```bash
git clone https://bitbucket.org/eunjeon/mecab-ko.git
cd mecab-ko && ./autogen.sh && ./configure && make && make install

git clone https://bitbucket.org/eunjeon/mecab-ko-dic.git
cd mecab-ko-dic && ./autogen.sh && ./configure && make && make install
```

## Usage

### 1. Text Preprocessing
```python
from konlpy.tag import Mecab
from src.preprocess import clean_text, preprocess_documents

# Initialize MeCab tokenizer
mecab = Mecab()

# Process documents
preprocessed_docs = preprocess_documents(raw_documents, mecab)
```

### 2. Topic Modeling
```python
from src.modeling import train_ctm_model

# Train CTM model
ctm_model = train_ctm_model(preprocessed_docs, n_components=5)

# Get topics
topics = ctm_model.get_topics(10)
```

### 3. Similarity Analysis
```python
# Load SBERT model
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Compute similarities
similarities = compute_similarities(documents, reference_topics, model)
```

## Contributing
This is a research project conducted at Sogang University's Graduate School of Management of Technology. For questions or contributions, please contact:
- Research Advisor: Prof. Kyootai Lee
- Graduate School of MOT, Sogang University

## License
This project is licensed under the MIT License - see the LICENSE file for details.
