import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Define o diretório que contém o arquivo de texto e o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')

# Verifica se o armazenamento vetorial Chroma já existe
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Garante que o arquivo de texto existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    # Lê o conteúdo de texto do arquivo
    loader = TextLoader(file_path)
    documents = loader.load()

    # Divide o documento em partes
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Exibe informações sobre os documentos divididos
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Cria embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OllamaEmbeddings(model="llama3")  # Atualize para um modelo de embedding válido, se necessário
    print("\n--- Finished creating embeddings ---")

    # Cria o armazenamento vetorial e persiste automaticamente
    print("\n--- Creating vector store ---")

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persistent_directory
    )
    print("\n--- Finished creating vector store ---")
else:
    print("Vector store already exists. No need to initialize.")
