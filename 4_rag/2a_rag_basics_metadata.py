import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Define o diretório contendo os arquivos de texto e o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, 'books')
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, 'chroma_db_with_metadata')

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Verifica se o armazenamento vetorial Chroma já existe
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Garante que o diretório de livros exista
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # Lista todos os arquivos de texto no diretório
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]

    # Lê o conteúdo do texto de cada arquivo e o armazena com metadados
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Adiciona metadados a cada documento indicando sua origem
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Divide os documentos em partes
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Exibe informações sobre os documentos divididos
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Cria embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OllamaEmbeddings(model="llama3")
    print("\n--- Finished creating embeddings ---")

    # Cria e persiste o armazenamento vetorial
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
