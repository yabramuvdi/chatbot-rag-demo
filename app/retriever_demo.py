#%%

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS, DeepLake
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import os
import yaml

##### main paths
input_path = "../data/"
vectordb_path = "../vectordb/"

#### read key and parameters
with open("../APIkey.txt") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key

#%%

################

with open("params.yaml") as stream:
    params = yaml.safe_load(stream)

    # define the models for embeddings    
    embedding = OpenAIEmbeddings(model=params["embeddings_model"])

    # load vector database
    vectordb = DeepLake(dataset_path=vectordb_path, 
                        embedding_function=embedding,
                        read_only=True)

# %%
