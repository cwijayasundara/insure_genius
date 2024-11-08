import warnings
import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

vector_db_loc = "./chroma_db"

warnings.filterwarnings('ignore')
_ = load_dotenv()

embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

parser = LlamaParse(
    result_type="markdown",
    parsing_instruction="These are insurence documents including policy limits and underwriting limits",
)

file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(input_files=['docs/medical-uw-limits.pdf', 
                                               'docs/income-protection-and-budget-income-protection-key-features.pdf', 
                                               'docs/flexible-protection-plan-key-features.pdf'],
                                  file_extractor=file_extractor).load_data()

db = chromadb.PersistentClient(path=vector_db_loc)
chroma_collection = db.get_or_create_collection("policy_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Test : Query Data from the persisted index
# query_engine = index.as_query_engine()
# response = query_engine.query("Whats the fpp critical illness underwriting limits for 51-55 olds?") 
# print(response)