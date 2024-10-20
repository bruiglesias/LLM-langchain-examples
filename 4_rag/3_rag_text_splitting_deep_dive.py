import os
from typing import List
from langchain.text_splitter import(
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
)

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Define o diretório que contém o arquivo de texto
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

# Verifica se o arquivo de texto existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

# Lê o conteúdo do texto do arquivo
loader = TextLoader(file_path)
documents = loader.load()

# Define o modelo de embedding
embeddings = OllamaEmbeddings(model="llama3")

# Função para criar e persistir o vetor de armazenamento
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")

# 1. Divisão baseada em caracteres
# Divide o texto em partes com base em um número especificado de caracteres.
# Útil para tamanhos de partes consistentes, independentemente da estrutura do conteúdo.
print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# 2. Divisão baseada em sentenças
# Divide o texto em partes com base em sentenças, garantindo que as partes terminem nos limites das sentenças.
# Ideal para manter a coerência semântica dentro das partes.
print("\n--- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")

# 3. Divisão baseada em tokens
# Divide o texto em partes com base em tokens (palavras ou subpalavras), usando tokenizers como o GPT-2.
# Útil para modelos de transformadores com limites rígidos de tokens.
print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, 'chroma_db_token')

# 4. Divisão recursiva baseada em caracteres
# Tenta dividir o texto em limites naturais (sentenças, parágrafos) dentro do limite de caracteres.
# Equilibra entre manter a coerência e obedecer aos limites de caracteres.
print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, 'chroma_db_rec_char')

# 5. Divisão personalizada
# Permite criar lógica de divisão personalizada com base em requisitos específicos.
# Útil para documentos com estrutura única que os divisores padrão não conseguem lidar.
class CustomTextSpliter(TextSplitter):
    def split_text(self, text):
        # Lógica personalizada para dividir o texto
        return text.split("\n\n")  # Exemplo: dividir por parágrafos
    
custom_splitter = CustomTextSpliter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, 'chroma_db_custom')

# Função para consultar um vetor de armazenamento
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
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

# Consulta cada vetor de armazenamento
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)
