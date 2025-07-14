import os
from dotenv import load_dotenv

load_dotenv()

langsmith_tracing=os.getenv("LANGCHAIN_TRACING")
langsmith_api_key=os.getenv("LANGCHAIN_API_KEY")
langsmith_project=os.getenv("LANGCHAIN_PROJECT")
langsmith_endpoint=os.getenv("LANGCHAIN_ENDPOINT")
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4.1-nano", api_key=openai_api_key)

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(web_paths=["https://medium.com/@jayanthsattineni/improving-signup-conversion-using-machine-learning-and-a-b-testing-1a75b04361d9/"])

docs = loader.load()

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "write a concise summary of the following text:\n{context}")
])

chain=create_stuff_documents_chain(llm,prompt) #init chain

result=chain.invoke({"context":docs})
print(result)
    

