import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Define o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define o modelo de embedding
embeddings = OllamaEmbeddings(model="llama3")

# Carrega o armazenamento vetorial existente com a função de embedding
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define a pergunta do usuário
query = "How did Juliet die?"

# Recupera documentos relevantes com base na consulta
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)
relevant_docs = retriever.invoke(query)

# Exibe os resultados relevantes com metadados
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")
