from dotenv import load_dotenv, find_dotenv
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import SystemMessage
from custom_tools import CreateEmailDraftTool, GenerateEmailResponseTool, ReplyEmailTool, EscalateTool, ProspectResearchTool, CategoriseEmailTool
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

import os
import requests
from google.cloud import pubsub_v1
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from email.message import EmailMessage

app = FastAPI()

load_dotenv(find_dotenv())
llm = ChatOpenAI(temperature=0, model="gpt-4-0613")

# Environment variable for project ID
project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
if not project_id:
  raise EnvironmentError(
      "The GOOGLE_CLOUD_PROJECT environment variable is not set.")

# OAuth2 Setup for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
creds = flow.run_local_server()  # Replit handles the port configuration

# Connect to the Gmail API
service = build('gmail', 'v1', credentials=creds)

# Pub/Sub Publisher Setup
publisher = pubsub_v1.PublisherClient()
topic_name = f'projects/{project_id}/topics/Mail'

# Create the Pub/Sub topic
try:
  publisher.create_topic(name=topic_name)
except Exception as e:
  print(f"Failed to create topic: {e}")

# Set up a watch on the Gmail inbox
try:
  request = {'labelIds': ['INBOX'], 'topicName': topic_name}
  service.users().watch(userId='me', body=request).execute()
except Exception as e:
  print(f"Failed to set up watch on Gmail: {e}")

# Pub/Sub Subscriber Setup
subscription_name = f'projects/{project_id}/subscriptions/MailSub'

with pubsub_v1.SubscriberClient() as subscriber:
  try:
    subscriber.create_subscription(name=subscription_name, topic=topic_name)
    # Add callback function and subscription logic here
  except Exception as e:
    print(f"Failed to create subscription: {e}")


class Message(BaseModel):
  subject: str
  sender: EmailStr
  body: str


@app.post("/")
def callback(message: Message):
  system_message = SystemMessage(content="""
          You are an email inbox assistant of Ale who is an electrical engineer working
          for a power electronics company. 
          Your goal is to handle all the incoming emails by categorising them based on 
          guideline and decide on next steps
          """)

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
  memory = ConversationSummaryBufferMemory(memory_key="memory",
                                           return_messages=True,
                                           llm=llm,
                                           max_token_limit=1000)

  agent = initialize_agent(
      tools,
      llm,
      agent=AgentType.OPENAI_FUNCTIONS,
      verbose=True,
      agent_kwargs=agent_kwargs,
      memory=memory,
  )

  # input = f"""
  #   New email received:
  #   {message.body}
  #   """

  # Call the agent function with the provided email content
  result = agent(message)

  # print the result as part of the response
  return result
