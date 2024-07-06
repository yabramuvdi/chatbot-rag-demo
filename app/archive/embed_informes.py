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

import os
import pandas as pd
from PyPDF2 import PdfReader 

input_path = "../data/informes/"
with open("../APIkey.txt") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key

#%%

#===================
# 0. Text loading
#===================

#### load a directory through LangChain classes
loader = DirectoryLoader(input_path, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
documents   # keeps the information on page and source document as metadata

#%%

# split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
texts

#%%

# define the models for embeddings (this does not embed the documents yet)
embeddings = OpenAIEmbeddings(document_model_name="text-embedding-ada-002",
                              query_model_name="text-embedding-ada-002")

# create a vector database with Chroma (could also use FAISS)
db = Chroma.from_documents(documents=texts, 
                           embedding=embeddings,
                           persist_directory="../vectordbs/informes/")

# save the database
db.persist()

# load from memory
#db = Chroma(persist_directory="./index/", embedding_function=embeddings)


#%%

prompt_template = """Actua como un experto en mercado. Usa las siguientes piezas de contexto para responder a la pregunta al final. Si no sabes la respuesta, simplemente dí que no sabes, no intentes inventar una respuesta.

{context}

Pregunta: {question}
Respuesta:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

#%%

# create a QA chain
# qa_chain = load_qa_chain(OpenAI(temperature=0,
#                                 model_name="text-davinci-003"
#                                 ), 
#                          chain_type="stuff",
#                          prompt=PROMPT)

qa_chain = load_qa_chain(ChatOpenAI(temperature=0,
                                    model_name="gpt-3.5-turbo"
                                    ), 
                         chain_type="stuff",
                         prompt=PROMPT)

qa = VectorDBQA(combine_documents_chain=qa_chain, 
                vectorstore=db, 
                return_source_documents=True,
                k=10,
                )

#%%

query = "Cuál es la comida eje para los ticos?"

# without sources
#qa.run(query)

# with sources
result = qa({"query": query})
print(result["result"])
print("============")
#%%

print("Fuentes dentro del informe:")
for doc in result["source_documents"]:
    print(doc.page_content)
    print(doc.metadata["source"], doc.metadata["page"])

#%%
