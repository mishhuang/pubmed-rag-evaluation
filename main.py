import os
from typing import List
from getpass import getpass
from dotenv import load_dotenv

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from anthropic_chat_generator import AnthropicChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

dataset = load_dataset("vblagoje/PubMedQA_instruction", split="train")
dataset = dataset.select(range(1000))
all_documents = [Document(content=doc["context"]) for doc in dataset]
all_questions = [doc["instruction"] for doc in dataset]
all_ground_truth_answers = [doc["response"] for doc in dataset]

document_store = InMemoryDocumentStore()

document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

indexing = Pipeline()
indexing.add_component(instance=document_embedder, name="document_embedder")
indexing.add_component(instance=document_writer, name="document_writer")

indexing.connect("document_embedder.documents", "document_writer.documents")

indexing.run({"document_embedder": {"documents": all_documents}})

# Load environment variables from .env file
load_dotenv()

# Set up Anthropic API key
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass("Enter Anthropic API key: ")

# Create RAG pipeline template
template = [
    ChatMessage.from_user(
        """
        You have to answer the following question based on the given context information only.
        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        """
    )
]

# Build RAG pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component(
    "query_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
rag_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=3))
rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
rag_pipeline.add_component("generator", AnthropicChatGenerator(model="claude-sonnet-4-5-20250929"))
rag_pipeline.add_component("answer_builder", AnswerBuilder())

rag_pipeline.connect("query_embedder", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.messages")
rag_pipeline.connect("generator.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")