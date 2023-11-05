import os
from dotenv import load_dotenv, find_dotenv
import openai
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import SystemMessage
from custom_tools import CreateEmailDraftTool, GenerateEmailResponseTool, ReplyEmailTool, EscalateTool, ProspectResearchTool, CategoriseEmailTool
from fastapi import FastAPI
from pydantic import BaseModel



app = FastAPI()
load_dotenv(find_dotenv())
openai.api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, model="gpt-4-0613")


system_message = SystemMessage(
    content="""
    You are an email inbox assistant of Ale who is an electrical engineer working
    for a power electronics company. 
    Your goal is to handle all the incoming emails by categorising them based on 
    guideline and decide on next steps
    """
)

tools = [
    CategoriseEmailTool(),
    ProspectResearchTool(),
    EscalateTool(),
    ReplyEmailTool(),
    CreateEmailDraftTool(),
    GenerateEmailResponseTool(),
]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


@app.post("/")
def analyse_email():

    test_email = """
    Hi, I would like to propose you a collaboration. I have 1 million investment funds. My name is Mind the Bridge
    and I am based in SF. Are you interested?

    """

    agent({"input": test_email})

