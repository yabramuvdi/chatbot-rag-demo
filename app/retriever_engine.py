if __name__ == "__main__":

    print("Starting QA system for Transcripciones...")

    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import DeepLake
    from langchain.prompts import PromptTemplate
    from dotenv import load_dotenv
    import os
    import pandas as pd

    import argparse

    # define global variables
    input_path = "../data/transcripciones/limpias/"

    # Add argparse section to accept command line arguments
    parser = argparse.ArgumentParser(description="Run QA on a file with specified parameters.")
    parser.add_argument("gen_text", help="Wheter to generate text or not to answer the question")
    parser.add_argument("analysis_type", help="Type of analysis to perform")
    parser.add_argument("analysis_selected", help="Specific analysis selected")
    parser.add_argument("persist_directory", help="Path to the directory where the index is persisted.")
    parser.add_argument("output_path", help="Path to the output directory where the results are stored.")
    parser.add_argument("user_input", help="User input to the QA system.")
    parser.add_argument("k_sources", help="Number of sources to use in the QA system.")

    # load arguments
    args = parser.parse_args()
    gen_text = args.gen_text
    analysis_type = args.analysis_type
    analysis_selected = args.analysis_selected
    persist_directory = args.persist_directory
    user_input = args.user_input
    K = int(args.k_sources)

    print(analysis_type)
    print(analysis_selected)

    # # [LOCAL ONLY] load API key locally from file
    # with open("./APIkey.txt") as f:
    #     api_key = f.read().strip()
    # # send to bash 
    # os.environ["OPENAI_API_KEY"] = api_key

    ## [HEROKU ONLY] read API key from the environment variable
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    # define the models for embeddings    
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    # load vector database
    #vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    #vectordb = FAISS.load_local(persist_directory, embeddings=embedding)
    vectordb = DeepLake(dataset_path=persist_directory, 
                        embedding_function=embedding,
                        read_only=True)
    
    # define generic prompt
    prompt_template = """Actua como un experto en mercadeo. Usa las siguientes piezas de contexto para responder a la pregunta al final. Si no sabes la respuesta, simplemente dí que no sabes, no intentes inventar una respuesta.

    PREGUNTA: {question}
    =========
    {summaries}
    =========
    
    RESPUESTA:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )

    #=================
    # Conditional source retrival
    #=================

    # create a retriever with the appropriate source filter and number of sources
    if analysis_type == "Todo":
        retriever = vectordb.as_retriever(search_kwargs={"k": K,
                                                         "distance_metric": "cos"})
        
    elif analysis_type == "Entrevista":
        # in this case "analysis_selected" will be the name of the file
        file_name = analysis_selected

        #### chromadb syntax
        # retriever = vectordb.as_retriever(search_kwargs={"k": K,
        #                                                  "filter": {"source": f"{input_path}{file_name}"}})
        
        #### DeepLake syntax
        # Define a custom filter function based on metadata
        # https://docs.deeplake.ai/en/latest/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.filter
        # https://docs.activeloop.ai/tutorials/deep-lake-vector-store-in-langchain
        def metadata_filter(x):
            # filter based on file name
            metadata = x['metadata'].data()['value']
            return f"{input_path}{file_name}" in metadata["source"]
        
        # initialize the retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": K,
                                                         "distance_metric": "cos"})
        
        # add filtering
        retriever.search_kwargs["filter"] = metadata_filter

    elif analysis_type == "Ciudad":
        city = analysis_selected
        print(f"Analisis para: {city}")
        
        def city_filter(x):
            # filter based on file name
            metadata = x['metadata'].data()['value']
            return f"{city}" == metadata["city"] 
    
        # initialize the retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": K,
                                                        "distance_metric": "cos"})
        
        # add filtering
        retriever.search_kwargs["filter"] = city_filter

    elif analysis_type == "Grupo":
        grupo = analysis_selected
        
        # if grupo == "Consumidor todos":
        #     def group_filter(x):
        #         # filter based on file name
        #         metadata = x['metadata'].data()['value']
        #         return f"Consumidor" in metadata["group"]
        # else:
        def group_filter(x):
            # filter based on file name
            metadata = x['metadata'].data()['value']
            return f"{grupo}" == metadata["group"]
        
        # initialize the retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": K,
                                                         "distance_metric": "cos"})
        # add filtering
        retriever.search_kwargs["filter"] = group_filter
    
    # elif analysis_type == "Ciudad - Grupo":
        
    #     ciudad_grupo = analysis_selected
    #     ciudad = ciudad_grupo.split("-")[0].strip()
    #     grupo = ciudad_grupo.split("-")[1].strip()

    #     def city_group_filter(x):
    #         # filter based on file name
    #         metadata = x['metadata'].data()['value']
    #         return f"{grupo}" in metadata["group"] and f"{ciudad}" in metadata["city"] 
        
    #     # initialize the retriever
    #     retriever = vectordb.as_retriever(search_kwargs={"k": K,
    #                                                      "distance_metric": "cos"})
    #     # add filtering
    #     retriever.search_kwargs["filter"] = city_group_filter

    ##### Retireve the most relevant documents
    docs = retriever.get_relevant_documents(user_input)
    print("Databased loaded and documents retrieved")
    
    # save the sources as a dataframe
    df_sources = pd.DataFrame(columns=["source_doc", "duration", "content"])
    for doc in docs:
        content = doc.page_content
        source = doc.metadata["source"]
        if "duration" in doc.metadata.keys():
            duration = doc.metadata["duration"]
        else:
            duration = ""
        
        df_sources = pd.concat([df_sources, pd.DataFrame({"source_doc": [source], "duration": [duration], "content": [content]})], ignore_index=True)

    # save file with sources
    df_sources.to_csv(args.output_path + "sources.csv", index=False)