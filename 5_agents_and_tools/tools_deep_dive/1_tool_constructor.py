# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

# Importa as bibliotecas necessárias
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool, Tool
from langchain_ollama import ChatOllama


# Funções para as ferramentas
def greet_user(name: str) -> str:
    """Cumprimenta o usuário pelo nome."""
    return f"Hello, {name}!"


def reverse_string(text: str) -> str:
    """Inverte a string fornecida."""
    return text[::-1]


def concatenate_strings(a: str, b: str) -> str:
    """Concatena duas strings."""
    return a + b


# Modelo Pydantic para argumentos das ferramentas
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="Primeira string")
    b: str = Field(description="Segunda string")


# Cria ferramentas usando a abordagem de construção com Tool e StructuredTool
tools = [
    # Use Tool para funções mais simples com um único parâmetro de entrada.
    # Isso é direto e não requer um esquema de entrada.
    Tool(
        name="GreetUser",  # Nome da ferramenta
        func=greet_user,  # Função a ser executada
        description="Cumprimenta o usuário pelo nome.",  # Descrição da ferramenta
    ),
    # Use Tool para outra função simples com um único parâmetro de entrada.
    Tool(
        name="ReverseString",  # Nome da ferramenta
        func=reverse_string,  # Função a ser executada
        description="Inverte a string fornecida.",  # Descrição da ferramenta
    ),
    # Use StructuredTool para funções mais complexas que exigem vários parâmetros de entrada.
    # StructuredTool permite definir um esquema de entrada usando Pydantic, garantindo validação e descrição adequadas.
    StructuredTool.from_function(
        func=concatenate_strings,  # Função a ser executada
        name="ConcatenateStrings",  # Nome da ferramenta
        description="Concatena duas strings.",  # Descrição da ferramenta
        args_schema=ConcatenateStringsArgs,  # Esquema que define os argumentos de entrada da ferramenta
    ),
]

# Inicializa um modelo ChatOllama
llm = ChatOllama(model="llama3")

# Puxa o template de prompt do hub
prompt = hub.pull("hwchase17/react")

# Cria o agente ReAct usando a função create_tool_calling_agent
agent = create_tool_calling_agent(
    llm=llm,  # Modelo de linguagem a ser usado
    tools=tools,  # Lista de ferramentas disponíveis para o agente
    prompt=prompt,  # Template de prompt para guiar as respostas do agente
)

# Cria o executor do agente
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,  # O agente a ser executado
    tools=tools,  # Lista de ferramentas disponíveis para o agente
    verbose=True,  # Habilita o registro detalhado
    handle_parsing_errors=True,  # Lida com erros de parsing de forma tranquila
)

# Testa o agente com consultas de exemplo
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)