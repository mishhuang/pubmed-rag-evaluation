"""Evaluation script for the RAG pipeline"""

import random
from main import rag_pipeline, all_questions, all_ground_truth_answers, all_documents
from haystack import Pipeline
from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
from haystack.components.evaluators.sas_evaluator import SASEvaluator
from anthropic_chat_generator import AnthropicChatGenerator

# Select 25 random questions and their corresponding ground truth answers and documents
questions, ground_truth_answers, ground_truth_docs = zip(
    *random.sample(list(zip(all_questions, all_ground_truth_answers, all_documents)), 25)
)

# Run the RAG pipeline on all questions and collect answers and retrieved documents
rag_answers = []
retrieved_docs = []

print("Running RAG pipeline on 25 questions...\n")
for i, question in enumerate(list(questions), 1):
    print(f"[{i}/25] Processing question...")
    response = rag_pipeline.run(
        {
            "query_embedder": {"text": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
    )
    
    rag_answers.append(response["answer_builder"]["answers"][0].data)
    retrieved_docs.append(response["answer_builder"]["answers"][0].documents)
    
    print(f"Question: {question[:100]}...")
    print(f"Answer: {rag_answers[-1][:100]}...")
    print("\n" + "-" * 80 + "\n")

# Build evaluation pipeline
print("Setting up evaluation pipeline...\n")
eval_pipeline = Pipeline()
eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
# FaithfulnessEvaluator needs a generator with json_mode=True to return JSON-formatted responses
eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator(chat_generator=AnthropicChatGenerator(model="claude-sonnet-4-5-20250929", json_mode=True)))
eval_pipeline.add_component("sas_evaluator", SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2"))

# Run evaluation
print("Running evaluation metrics...\n")
results = eval_pipeline.run(
    {
        "doc_mrr_evaluator": {
            "ground_truth_documents": list([d] for d in ground_truth_docs),
            "retrieved_documents": retrieved_docs,
        },
        "faithfulness": {
            "questions": list(questions),
            "contexts": list([d.content] for d in ground_truth_docs),
            "predicted_answers": rag_answers,
        },
        "sas_evaluator": {
            "predicted_answers": rag_answers,
            "ground_truth_answers": list(ground_truth_answers)
        },
    }
)

# Print evaluation results
print("=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)
print(f"\nDocument Mean Reciprocal Rank (MRR): {results['doc_mrr_evaluator']['score']}")
print(f"Faithfulness Score: {results['faithfulness']['score']}")
print(f"Semantic Answer Similarity (SAS): {results['sas_evaluator']['score']}")
print("\n" + "=" * 80)

