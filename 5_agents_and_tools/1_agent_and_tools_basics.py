from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (AgentExecutor, create_react_agent)
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Define uma função de ferramenta muito simples que retorna a hora atual
def get_current_time(*args, **kwargs):
    """Retorna a hora atual no formato H:MM AM/PM."""
    import datetime  # Importa o módulo datetime para obter a hora atual

    now = datetime.datetime.now()  # Obtém a hora atual
    return now.strftime("%I:%M %p")  # Formata a hora no formato H:MM AM/PM

# Lista de ferramentas disponíveis para o agente
tools = [
    Tool(
        name="Time",  # Nome da ferramenta
        func=get_current_time,  # Função que a ferramenta irá executar
        # Descrição da ferramenta
        description="Útil quando você precisa saber a hora atual",
    ),
]

# Puxa o template de prompt do hub
# ReAct = Reason and Action (Raciocínio e Ação)
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# Inicializa um modelo ChatOllama
llm = ChatOllama(model="llama3")

# Cria o agente ReAct usando a função create_react_agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Cria um executor de agente a partir do agente e das ferramentas
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Executa o agente com uma consulta de teste
response = agent_executor.invoke({"input": "What time is it?"})

# Imprime a resposta do agente
print("response:", response)
