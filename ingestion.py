import asyncio
import os
import ssl
import certifi
from typing import Dict, List, Any
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_info, log_header, log_error, log_success, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNBDLE"] = certifi.where()

embeddings = AzureOpenAIEmbeddings(
    model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10
)

chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
pinecone_vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
tavily_crawl = TavilyCrawl()
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)

def chunk_urls(urls: List[str], chunk_size: int = 20)-> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i: i+chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        log_info(f"** TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs", color=Colors.BLUE)
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(f"** TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents")
        return docs
    except Exception as e:
        log_error(f"** TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []
    
async def async_extract(url_batches: List[List[str]]):
    """Document Extraction from batches"""
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(f"** TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches", color=Colors.DARKCYAN)
    
    tasks = [extract_batch(batch, i + 1)for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    #Filter out exceptions and flatten result
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"** TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extract_page in result["results"]:
                document = Document(
                    page_content=str(extract_page["raw_content"]),
                    metadata = {"source": extract_page["url"]}
                )
                all_pages.append(document)
    log_success(f"** TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}")

    if failed_batches > 0:
        log_warning(f"** TavilyExtract: {failed_batches} batches faied during extraction")
    
    return all_pages

# ==========================================================================
# IMPLEMENTATION 1: Web scrapping with TavilyCrawl 
# ==========================================================================
def web_scrapping_with_tavily_crawl():
    """Crawl documentation from Site using TavilyCrawl"""
    log_header("DOCUMENT INGESTION PIPELINE")
    log_info("** TavilyCrawl: Start the crawl documentation from https://python.langchain.com/", color=Colors.CYAN)

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,
        "extract_depth": "advanced"
    })
    all_docs = [Document(page_content=str(result['raw_content']), metadata={"source": result['url']}) for result in res["results"]]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site.")

# ==========================================================================
# IMPLEMENTATION 2: Web scrapping with TavilyMap and TavilyExtract 
# ==========================================================================
async def web_scrapping_with_tavily_map_and_extract():
    """Crawl documentation from Site using TavilyMap and TavilyExtract"""
    log_header("DOCUMENT INGESTION PIPELINE")
    log_info("** TavilyMap: Start to map documentation structure from https://python.langchain.com/", color=Colors.PURPLE)

    site_map = tavily_map.invoke("https://python.langchain.com/")
   
    log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site.")

    # Split URLs into batches of 20
    url_batches = chunk_urls(site_map["results"], chunk_size=20)
    log_info(f"** URL Processing: Split {len(site_map)} URLs into {len(url_batches)} batches", color=Colors.BLUE)

    # Extract documents from URLs
    all_docs = await async_extract(url_batches)
    print("Extraction completed")

async def main(): 
    """Main async function to orchestrator the entire process"""
    
    # Option 1: Crawl documentation by using TavilyCrawl
    web_scrapping_with_tavily_crawl()

    # Option 2: Crawl documentation by using TavilyMap and TavilyExtract
    await web_scrapping_with_tavily_map_and_extract()

if __name__ == "__main__":
    asyncio.run(main())