# pip install ragas openai

import inspect
import os

import pandas as pd
from openai import AsyncOpenAI

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

# RAGAS metrics (collections API)
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy
)

from ragas.embeddings import OpenAIEmbeddings


# =========================================================
# 1. OPENAI API KEY
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else AsyncOpenAI()


# =========================================================
# 2. RAGAS EVALUATION LLM
# =========================================================

from ragas.llms import llm_factory

evaluator_llm = llm_factory(
    "gpt-4o-mini",
    client=client
)


# =========================================================
# 3. EMBEDDING MODEL
# =========================================================

evaluator_embeddings = OpenAIEmbeddings(
    client=client,
    model="text-embedding-3-small"
)


# =========================================================
# 4. TEST DATA
# =========================================================

samples = [

    SingleTurnSample(
        user_input="What is a vector database?",

        retrieved_contexts=[
            "A vector database stores embeddings of documents "
            "and supports similarity search.",

            "Vector databases are commonly used in RAG systems "
            "to retrieve semantically relevant information."
        ],

        response=(
            "A vector database stores embeddings and allows "
            "semantic similarity search to retrieve relevant information."
        ),

        reference=(
            "A vector database stores document embeddings and "
            "performs similarity search to retrieve semantically "
            "relevant information."
        )
    ),

    SingleTurnSample(
        user_input="What is RAG?",

        retrieved_contexts=[
            "Retrieval-Augmented Generation combines document "
            "retrieval with language generation.",

            "Relevant documents are retrieved and provided to "
            "the language model as context."
        ],

        response=(
            "RAG retrieves relevant information from a knowledge "
            "base and provides it to an LLM as context for generating "
            "an answer."
        ),

        reference=(
            "RAG retrieves relevant information from an external "
            "knowledge source and provides it to an LLM as context "
            "before generating an answer."
        )
    )
]


# =========================================================
# 5. CREATE DATASET
# =========================================================

dataset = EvaluationDataset(
    samples=samples
)


# =========================================================
# 6. CREATE METRICS
# =========================================================

metrics = [

    ContextPrecision(
        llm=evaluator_llm
    ),

    ContextRecall(
        llm=evaluator_llm
    ),

    Faithfulness(
        llm=evaluator_llm
    ),

    AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
]


# =========================================================
# 7. RUN RAGAS
# =========================================================

rows = []

for sample in dataset.samples:

    row = sample.model_dump(exclude_none=True)

    for metric in metrics:
        # Pass only the fields this metric needs
        params = inspect.signature(metric.ascore).parameters
        inputs = {name: getattr(sample, name) for name in params}

        row[metric.name] = metric.score(**inputs).value

    rows.append(row)

df = pd.DataFrame(rows)

result = {metric.name: df[metric.name].mean() for metric in metrics}


# =========================================================
# 8. DISPLAY RESULTS
# =========================================================

print()
print("========================================")
print("       RAGAS EVALUATION RESULTS")
print("========================================")
print()

print(result)

print()
print("========================================")
print("          RESULTS DATAFRAME")
print("========================================")
print()

# Show all rows, columns and full cell text
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

print(df)