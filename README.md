# PubMed RAG Evaluation

A Retrieval-Augmented Generation (RAG) pipeline for evaluating medical question-answering using the PubMedQA dataset. This project uses Haystack, Anthropic Claude, and sentence transformers to build and evaluate a RAG system.

## Features

- **Document Indexing**: Indexes PubMedQA documents using sentence transformers embeddings
- **RAG Pipeline**: Retrieves relevant documents and generates answers using Anthropic Claude
- **Comprehensive Evaluation**: Evaluates the pipeline using three metrics:
  - **Document Mean Reciprocal Rank (MRR)**: Measures retrieval quality
  - **Faithfulness**: Evaluates if generated answers can be inferred from retrieved contexts
  - **Semantic Answer Similarity (SAS)**: Measures semantic similarity between predicted and ground truth answers
- **Evaluation Reports**: Generates aggregated and detailed reports with DataFrame output for analysis

## Project Structure

```
pubmed-rag-evaluation/
├── main.py                          # Main script: data loading, indexing, and RAG pipeline setup
├── evaluate.py                      # Evaluation script: runs evaluation on 25 questions
├── anthropic_chat_generator.py      # Custom Anthropic Claude generator component
├── requirements.txt                  # Python dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pubmed-rag-evaluation
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### API Keys

This project requires an Anthropic API key for the Claude model. Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

The `.env` file is already in `.gitignore` to keep your API key secure.

## Usage

### 1. Set Up the Pipeline

First, run `main.py` to load the dataset, index documents, and set up the RAG pipeline:

```bash
python main.py
```

This script will:
- Load the PubMedQA dataset (1000 samples)
- Create document embeddings using sentence transformers
- Index documents into an InMemoryDocumentStore
- Set up the RAG pipeline with Anthropic Claude

### 2. Run Evaluation

After setting up the pipeline, run the evaluation script:

```bash
python evaluate.py
```

This script will:
- Select 25 random questions from the dataset
- Run the RAG pipeline on each question
- Evaluate using all three metrics (MRR, Faithfulness, SAS)
- Generate and display:
  - Basic evaluation results
  - Aggregated report with all metric scores
  - Detailed DataFrame report with per-sample scores
  - Top 3 and bottom 3 SAS score comparisons

### Example Output

The evaluation script produces:
- **Evaluation Results**: Summary scores for all metrics
- **Aggregated Report**: Overall performance metrics
- **Detailed Report**: DataFrame with scores for each question
- **Top/Bottom Performers**: Best and worst performing examples by SAS score

## Evaluation Metrics

### Document Mean Reciprocal Rank (MRR)
Measures how well the retrieval system finds relevant documents. Higher scores indicate better retrieval performance.

### Faithfulness
Evaluates whether the generated answer can be inferred from the retrieved context documents. Uses an LLM (Claude) to assess answer faithfulness.

### Semantic Answer Similarity (SAS)
Measures the semantic similarity between predicted answers and ground truth answers using sentence transformers. Scores range from 0 to 1, with higher scores indicating better semantic alignment.

## Custom Components

### AnthropicChatGenerator

A custom Haystack component that integrates Anthropic's Claude API. Features:
- Support for plain text responses (default, for RAG pipeline)
- JSON mode support (for evaluators that require structured output)
- Automatic API key management from environment variables

## Requirements

- Python 3.12+
- haystack-ai
- datasets>=2.6.1
- sentence-transformers>=4.1.0
- numpy<2
- python-dotenv
- anthropic
- pandas (installed as dependency)

## Notes

- The indexing process may take a few minutes on first run
- Evaluation on 25 questions takes approximately 5-10 minutes depending on API response times
- The InMemoryDocumentStore is used for simplicity but doesn't scale well for production systems

