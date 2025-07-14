import os
from dotenv import load_dotenv
from pydantic import SecretStr
import asyncio

load_dotenv()

os.environ["USER_AGENT"] = "LangChain-WebScraper/1.0"

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is not None:
    openai_api_key = SecretStr(openai_api_key)
else:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(web_paths=["https://medium.com/@jayanthsattineni/improving-signup-conversion-using-machine-learning-and-a-b-testing-1a75b04361d9/"])

docs = loader.load()

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

map_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a concise summary of the following text:\n\n{text}")
])

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "The following is a set of summaries:\n\n{summaries}\n\nTake these and distill it into a final, consolidated summary of the main themes.")
])

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
split_docs = text_splitter.split_documents(docs)
print(f"Generated {len(split_docs)} document chunks.")

async def map_summarize(text: str) -> str:
    """Map function: summarize a single chunk"""
    prompt = map_prompt.invoke({"text": text})
    response = await llm.ainvoke(prompt)
    return str(response.content)

async def reduce_summaries(summaries: list[str]) -> str:
    """Reduce function: combine multiple summaries into one"""
    combined_summaries = "\n\n".join(summaries)
    prompt = reduce_prompt.invoke({"summaries": combined_summaries})
    response = await llm.ainvoke(prompt)
    return str(response.content)

async def map_reduce_summarize():
    """Main map-reduce function"""
    print("Starting map-reduce summarization...")
    
    print("Map phase: summarizing individual chunks...")
    map_tasks = [map_summarize(doc.page_content) for doc in split_docs]
    chunk_summaries = await asyncio.gather(*map_tasks)
    
    print(f"Generated {len(chunk_summaries)} chunk summaries.")
    
    print("Reduce phase: combining summaries...")
    final_summary = await reduce_summaries(chunk_summaries)
    
    return final_summary

if __name__ == "__main__":
    result = asyncio.run(map_reduce_summarize())
    print("\n" + "="*50)
    print("FINAL SUMMARY (Map-Reduce)")
    print("="*50)
    print(result)