# https://python.langchain.com/en/latest/use_cases/tabular.html

#%%

from langchain.chains import VectorDBQA
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.agents import create_pandas_dataframe_agent

import os
import pandas as pd

input_path = "../data/bases/"
with open("../APIkey.txt") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key

#%%

#===================
# 0. Data loading
#===================

df = pd.read_excel(input_path + "Base Nueces Panamá 2022 - Ola 1.xlsx")
df

#%%

agent = create_pandas_dataframe_agent(OpenAI(temperature=0,
                                             model_name="text-davinci-003"), 
                                      df, 
                                      verbose=True)

#%%

agent.run("Could you give me the name of 10 random columns?")

