import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Define o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")


# Define o modelo de embedding
embeddings = OllamaEmbeddings(model="llama3")


# Carrega o vetor de armazenamento existente com os diferentes tipos de busca e 
# parâmetros
def query_vector_store(store_name, query, embedding_function, search_type, search_kwargs):
    
    if os.path.exists(persistent_directory):
        
        print(f"\n--- Querying the vector store {store_name} ---")
        
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function
        )
        
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        relevant_docs = retriever.invoke(query)

        # Exibe os resultados relevantes com metadados
        print(f"\n--- Relevant Documents for {store_name} ---")

        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")
    

# Define a pergunta do usuário
query = "How did Juliet die?"


# 1. Similarity Search
# Este método recupera documentos com base em vetor de similaridade.
# Ele encontra os documentos mais semelhantes ao vetor da consulta com base na similaridade cosseno.
# Use isso quando você quiser recuperar os k documentos mais semelhantes.
print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db_with_metadata", query, embeddings,
                   "similarity", {"k": 3})


# 2. Max Marginal Relevance (MMR)
# Este método equilibra a seleção de documentos que são relevantes para a consulta e diversos entre si.
# 'fetch_k' especifica o número de documentos a serem inicialmente buscados com base na similaridade.
# 'lambda_mult' controla a diversidade dos resultados: 1 para diversidade mínima, 0 para diversidade máxima.
# Use isso quando você quiser evitar redundância e recuperar documentos relevantes, mas variados.
# Nota: A relevância mede quão bem os documentos correspondem à consulta.
# Nota: A diversidade garante que os documentos recuperados não sejam muito semelhantes entre si,
#       proporcionando uma gama mais ampla de informações.
print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store("chroma_db_with_metadata", query, embeddings, 
                   "mmr", {"k": 3, "fetch_k": 20, "lambda_mult": 0.5})


# 3. Similarity Score Threshold
# Este método recupera documentos que superam um determinado limite de pontuação de similaridade.
# 'score_threshold' define a pontuação mínima de similaridade que um documento deve ter para ser considerado relevante.
# Use isso quando você quiser garantir que apenas documentos altamente relevantes sejam recuperados, filtrando os menos relevantes.
print("\n--- Using Similarity Score Threshold ---")
query_vector_store("chroma_db_with_metadata", query, embeddings, "similarity_score_threshold", 
                   {"k": 3, "score_threshold": 0.1},)


print("Querying demonstrations with different search types completed.")
