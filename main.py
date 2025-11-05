from datasets import load_dataset
from haystack import Document

dataset = load_dataset("vblagoje/PubMedQA_instruction", split="train")
dataset = dataset.select(range(1000))
all_documents = [Document(content=doc["context"]) for doc in dataset]
all_questions = [doc["instruction"] for doc in dataset]
all_ground_truth_answers = [doc["response"] for doc in dataset]