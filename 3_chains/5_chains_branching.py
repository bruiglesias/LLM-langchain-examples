from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_ollama import ChatOllama

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar um modelo ChatOllama
model = ChatOllama(model = "llama3")

# Definir templates de prompt para diferentes tipos de feedback
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}."),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

# Definir o template de classificação de feedback
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

# Definir as ramificações executáveis para lidar com feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()  # Cadeia de feedback positivo
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()  # Cadeia de feedback negativo
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()  # Cadeia de feedback neutro
    ),
    escalate_feedback_template | model | StrOutputParser()  # Cadeia para escalar feedback
)

# Criar a cadeia de classificação
classification_chain = classification_template | model | StrOutputParser()

# Combinar classificação e geração de resposta em uma única cadeia
chain = classification_chain | branches

# Executar a cadeia com um exemplo de revisão
# Bom review - "The product is excellent. I really enjoyed using it and found it very helpful."
# review ruim - "The product is terrible. It broke after just one use and the quality is very poor."
# review neutro - "The product is okay. It works as expected but nothing exceptional."
# Padrão - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "The product is terrible. It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback": review})

# Saída do resultado
print(result)
