import os
#from docx import Document
import ollama
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import pandas as pd



WORKING_DIR = "C:\\Users\\jmart\\Documents\\hackathon\\outi"


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)



rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="eden",

    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(texts, embed_model="nomic-embed-text:latest"),
    ),
)


# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=ollama_model_complete,
#     llm_model_name="qwen2m",
#
#     embedding_func=EmbeddingFunc(
#         embedding_dim=768,
#         max_token_size=8192,
#         func=lambda texts: ollama_embedding(texts, embed_model="nomic-embed-text:latest"),
#     ),
# )

print("Loading text")
df = pd.read_csv("C:\\Users\\jmart\\Documents\\hackathon\\Desert_Temp_RH_FEB-2025.csv")
rag.insert(df.to_string())




#print(rag.query("If I have a GPA of 2.85 do I meet Satisfactory Academic Progress for the graduate program?", param=QueryParam(mode="local", top_k=5)))
# Perform naive search
'''
print("Asking question 1-1...")
print(
    rag.query("If I have a GPA of 2.85 do I meet Satisfactory Academic Progress for the undergraduate program?", param=QueryParam(mode="hybrid"))
)

print("Asking question 1-2...")
print(
    rag.query("If I have a GPA of 2.85 do I meet Satisfactory Academic Progress for the graduate program?", param=QueryParam(mode="hybrid"))
)

print("Asking question 2...")
print(
    rag.query("What website should I visit to apply for Graduate Admissions?", param=QueryParam(mode="hybrid"))
)

print("Asking question 3...")
print(
    rag.query("What are the Graduate Admission Requirements?", param=QueryParam(mode="hybrid"))
)

print("Asking question 4...")
print(
    rag.query("What are some Electives for Software Engineering?", param=QueryParam(mode="hybrid"))
)

'''