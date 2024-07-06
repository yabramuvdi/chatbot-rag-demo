#%%

import os
import yaml
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.prompts import PromptTemplate
import pandas as pd

##### main paths
input_path = "../data/"
vectordb_path = "../vectordb/"
output_path = "./temp/"

#### read key and parameters
with open("../APIkey.txt") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key

with open("params.yaml") as stream:
    params = yaml.safe_load(stream)


#%%

################
# FILTERS
    
#### DeepLake syntax
# Define a custom filter function based on metadata
# https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.filter
# https://docs.activeloop.ai/tutorials/deep-lake-vector-store-in-langchain


def filename_filter(x):
    # filter based on file name
    metadata = x['metadata'].data()['value']
    return f"{file_name}" in metadata["source"]

def country_filter(x):
    # filter based on file name
    metadata = x['metadata'].data()['value']
    return f"{pais}" == metadata["country"]

#%%
    
# define the models for embeddings    
embedding = OpenAIEmbeddings(model=params["embeddings_model"])

# load vector database
vectordb = DeepLake(dataset_path=vectordb_path, 
                    embedding=embedding,
                    read_only=True)

# %%

#############

# define generic prompt
prompt_template = """Usa las siguientes piezas de contexto para responder a la pregunta al final. Si no sabes la respuesta, simplemente dí que no sabes, no intentes inventar una respuesta.

PREGUNTA: {question}
=========
{summaries}
=========

RESPUESTA:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
)

#%%

#=================
# Conditional source retrival
#=================

K = 5
analysis_type = "Episodio"
analysis_selected = "t13_e16"

retriever = vectordb.as_retriever(search_kwargs={"k": K,
                                                 "distance_metric": params["docs_distance_metric"]})

# create a retriever with the appropriate source filter and number of sources
if analysis_type == "Todo":
    pass

elif analysis_type == "Episodio":
    # in this case "analysis_selected" will be the name of the file
    file_name = analysis_selected
    # add filtering
    retriever.search_kwargs["filter"] = filename_filter

elif analysis_type == "Pais":
    pais = analysis_selected
    # add filtering
    retriever.search_kwargs["filter"] = country_filter

#%%
    
user_input = "Qué pasa en el episodio?"
    
##### Retireve the most relevant documents
docs = retriever.get_relevant_documents(user_input)
print("Databased loaded and documents retrieved")
# %%

# save the sources as a dataframe
df_sources = pd.DataFrame(columns=["source_doc", "content"])
for doc in docs:
    content = doc.page_content
    source = doc.metadata["episode_name"]
    df_sources = pd.concat([df_sources, pd.DataFrame({"source_doc": [source], "content": [content]})], ignore_index=True)

#%%
    
# save file with sources
df_sources.to_csv(output_path + "sources.csv", index=False)

# %%
