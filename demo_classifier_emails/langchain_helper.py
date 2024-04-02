from langchain.agents.agent_toolkits import GmailToolkit
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain



from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

toolkit = GmailToolkit()

llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key,
    )

agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)

result = agent.run("""Faca a leitura dos meus 5 ultimos emails e Classifique o conteudo de acordo com os seguintes assuntos:
                   'Pagamentos', 'Elogio', 'Informativo', 'Sugestão' e gere um draft contendo uma lista com o titulo e a classificação
                   de cada um deles""")

print(result)
