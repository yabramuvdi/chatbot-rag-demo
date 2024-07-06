if __name__ == "__main__":

    print("Starting QA system for Informes...")

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
    from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain


    import os
    import pandas as pd
    from PyPDF2 import PdfReader 

    import argparse

    # Add argparse section to accept command line arguments
    parser = argparse.ArgumentParser(description="Run QA on a file with specified parameters.")
    parser.add_argument("input_path", help="Path to the input directory containing the files.")
    parser.add_argument("file_name", help="Name of the file to interact with.")
    parser.add_argument("persist_directory", help="Path to the directory where the index is persisted.")
    parser.add_argument("output_path", help="Path to the output directory where the results are stored.")
    parser.add_argument("user_input", help="User input to the QA system.")

    # load arguments
    args = parser.parse_args()
    input_path = args.input_path
    file_name = args.file_name
    persist_directory = args.persist_directory
    user_input = args.user_input
    K = 5

    with open("../APIkey.txt") as f:
        api_key = f.read().strip()

    os.environ["OPENAI_API_KEY"] = api_key

    # define the models for embeddings
    embedding = OpenAIEmbeddings(document_model_name="text-embedding-ada-002",
                                    query_model_name="text-embedding-ada-002")

    # load vector database
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # define generic prompt
    prompt_template = """Actua como un experto en mercado. Usa las siguientes piezas de contexto para responder a la pregunta al final. Si no sabes la respuesta, simplemente dí que no sabes, no intentes inventar una respuesta.

    PREGUNTA: {question}
    =========
    {summaries}
    =========
    
    Respuesta:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )


    # create a retriever with the appropriate source filter
    if file_name == "All":
        retriever = vectordb.as_retriever(k=K)
    else:
        retriever = vectordb.as_retriever(k=K, search_kwargs={"filter": {"source": f"../data/informes/{file_name}"}})


    # create a QA chain
    qa_chain = load_qa_with_sources_chain(ChatOpenAI(temperature=0,
                                                     model_name="gpt-3.5-turbo"
                                                    ), 
                                          chain_type="stuff",
                                          prompt=PROMPT)

    chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, 
                                        retriever=retriever, 
                                        return_source_documents=True)

    print("Database loaded and QA chain with sources initialized.")

    # with sources
    result = chain({"question": user_input}, return_only_outputs=True)

    # save the result
    with open(args.output_path + "results.txt", "w") as f:
        f.write(result["answer"])
        print("File save with main answer")
    
    # save the sources as a dataframe
    df_sources = pd.DataFrame(columns=["source_doc", "page", "content"])
    for doc in result["source_documents"]:
        content = doc.page_content
        source = doc.metadata["source"]
        page= doc.metadata["page"]
        df_sources = df_sources.append({"source_doc": source, "page": page, "content": content}, ignore_index=True)

    df_sources.to_csv(args.output_path + "sources.csv", index=False)