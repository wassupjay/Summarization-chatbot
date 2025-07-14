import os
from dotenv import load_dotenv
from pydantic import SecretStr
import asyncio

load_dotenv()

os.environ["USER_AGENT"] = "LangChain-WebScraper/1.0"

openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(web_paths=["https://medium.com/@jayanthsattineni/improving-signup-conversion-using-machine-learning-and-a-b-testing-1a75b04361d9/"])

docs = loader.load()

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

stuff_prompt = ChatPromptTemplate.from_messages([
    ("system", "write a concise summary of the following text:\n{context}")
])

stuff_chain = create_stuff_documents_chain(llm, stuff_prompt)

# Map-reduce
map_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a concise summary of the following text:\n\n{text}")
])

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "The following is a set of summaries:\n\n{summaries}\n\nTake these and distill it into a final, consolidated summary of the main themes.")
])

# Splitin chunks for mapred
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
split_docs = text_splitter.split_documents(docs)

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
    # Map-summarize each chunk
    map_tasks = [map_summarize(doc.page_content) for doc in split_docs]
    chunk_summaries = await asyncio.gather(*map_tasks)
    
    # Reduce-combine summaries
    final_summary = await reduce_summaries(chunk_summaries)
    return final_summary

def stuff_summarize():
    """Stuff summarizer function"""
    return stuff_chain.invoke({"context": docs})

if __name__ == "__main__":
    print("="*60)
    print("COMPARING SUMMARIZATION APPROACHES")
    print("="*60)
    
    print(f"\nDocument chunks: {len(split_docs)}")
    print(f"Total documents: {len(docs)}")
    
    print("\n" + "="*60)
    print("STUFF SUMMARIZER (Single Pass)")
    print("="*60)
    stuff_result = stuff_summarize()
    print(stuff_result)
    
    print("\n" + "="*60)
    print("MAP-REDUCE SUMMARIZER (Multi-Pass)")
    print("="*60)
    map_reduce_result = asyncio.run(map_reduce_summarize())
    print(map_reduce_result)
    
    print("\n" + "="*60)
    print("COMPARISON NOTES")
    print("="*60)
    print("• Stuff Summarizer: Single pass, processes all content at once")
    print("• Map-Reduce Summarizer: Multi-pass, summarizes chunks then combines")
    print("• Map-Reduce is better for very long documents that exceed token limits")
    print("• Stuff Summarizer is simpler and faster for shorter documents") 

'''
    O/P:
============================================================
COMPARING SUMMARIZATION APPROACHES
============================================================

Document chunks: 1
Total documents: 1

============================================================
STUFF SUMMARIZER (Single Pass)
============================================================
Jayanth's blog discusses a project focused on enhancing signup conversion rates using machine learning and A/B testing. The project involves training a random forest model to predict user conversions and personalizing call-to-action messages based on the model's predictions. A simulated dataset of 10,000 user sessions is created to train the model, after which an A/B test is conducted to validate the effectiveness of personalization. Sattineni also emphasizes the importance of statistical validation for growth ideas, highlighting A/B testing as a key tool for assessing the impact of changes. Future steps involve applying real user data and deploying the model for real-time personalization.

============================================================
MAP-REDUCE SUMMARIZER (Multi-Pass)
============================================================
Jayanth's article highlights a project that leverages machine learning, specifically a Random Forest model, and A/B testing to improve signup conversion rates. The project involves predicting user conversions and personalizing call-to-action (CTA) messages based on user behavior. Key steps include data simulation, model training, CTA customization, and A/B testing to assess effectiveness, with results analyzed using statistical methods. The findings underscore the significance of A/B testing in confirming the success of personalization strategies. Recommendations for the future include utilizing real user data and establishing live dashboards for continuous performance monitoring.

============================================================
COMPARISON NOTES
============================================================
• Stuff Summarizer: Single pass, processes all content at once
• Map-Reduce Summarizer: Multi-pass, summarizes chunks then combines
• Map-Reduce is better for very long documents that exceed token limits
• Stuff Summarizer is simpler and faster for shorter documents
'''