import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Define o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')

# Define o modelo de embedding
embeddings = OllamaEmbeddings(model="llama3")

# Carrega o armazenamento vetorial existente com a função de embedding
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define a pergunta do usuário
query = "Who is Odysseus' wife?"

# Recupera documentos relevantes com base na consulta
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 3, 'score_threshold': 0.4}
)
relevant_docs = retriever.invoke(query)

# Exibe os resultados relevantes com metadados
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
