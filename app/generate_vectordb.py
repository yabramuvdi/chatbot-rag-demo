#%%

from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader, DirectoryLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS, DeepLake
from langchain_openai import OpenAIEmbeddings

import os
import pandas as pd
import numpy as np
import yaml

from langchain.indexes import VectorstoreIndexCreator


from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA


##### main paths
input_path = "../data/"
vectordb_path = "../vectordb/"

#### read key and parameters
with open("../APIkey.txt") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key

with open("params.yaml") as stream:
    params = yaml.safe_load(stream)


#%%

#===================
# 0. Text loading
#===================

all_files = os.listdir(input_path)
all_files = [f for f in all_files if "metadata.csv" not in f]
files_metadata = pd.read_csv(input_path + "metadata.csv", sep=";")
documents_txt = []
for file in all_files:
    # load
    loader = TextLoader(input_path + file)
    doc = loader.load()
    # add metadata
    file_data = files_metadata.loc[files_metadata["file_name"] == file.replace(".txt", "")]
    doc[0].metadata["source"] = file_data["file_name"].values[0]
    doc[0].metadata["episode_name"] = file_data["clean_name"].values[0]
    doc[0].metadata["country"] = file_data["country"].values[0]
    documents_txt.append(doc[0])

#%%

#===================
# 1. Chunkenization
#===================
    
# split documents into chunks
chunk_size = params["chunk_size"]
chunk_overlap = params["overlap"]

#text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                               chunk_overlap=chunk_overlap,
                                               length_function=len)

# split texts
processed_docs = text_splitter.split_documents(documents_txt)
print(f"Number of chunks from .txt files: {len(processed_docs )}")

#%%

#===================
# 2. Text embedding and Vector database
#===================

# define the models for embeddings (this does not embed the documents yet)
embeddings = OpenAIEmbeddings(model=params["embeddings_model"])
embeddings
#%%

#### DeepLake
# we will use it for its capacity to filter queries using metadata
# database is automatically saved to the path
db = DeepLake.from_documents(documents=processed_docs, 
                             embedding=embeddings,
                             dataset_path=vectordb_path)

#%%

# #==============
# # QA Chain demonstration
# #==============

# prompt_template = """Actua como un experto en mercado. Usa las siguientes piezas de contexto para responder a la pregunta al final. Si no sabes la respuesta, simplemente dí que no sabes, no intentes inventar una respuesta.

# {context}

# Pregunta: {question}
# Respuesta:"""

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )

# #%%

# # create a QA chain
# # qa_chain = load_qa_chain(OpenAI(temperature=0,
# #                                 model_name="text-davinci-003"
# #                                 ), 
# #                          chain_type="stuff",
# #                          prompt=PROMPT)

# qa_chain = load_qa_chain(ChatOpenAI(temperature=0,
#                                     model_name="gpt-3.5-turbo"
#                                     ), 
#                          chain_type="stuff",
#                          prompt=PROMPT)

# qa = VectorDBQA(combine_documents_chain=qa_chain, 
#                 vectorstore=db, 
#                 return_source_documents=True,
#                 k=10,
#                 )

# #%%

# query = "Cuál es la comida eje para los ticos?"

# # without sources
# #qa.run(query)

# # with sources
# result = qa({"query": query})
# print(result["result"])
# print("============")
# #%%

# print("Fuentes dentro del informe:")
# for doc in result["source_documents"]:
#     print(doc.page_content)
#     print(doc.metadata["source"], doc.metadata["page"])

# #%%
