import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
# from sentence_transformers import SentenceTransformer
import pickle
import time

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv()


st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")


main_placeholder = st.empty()
llm = ChatGroq(
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model = 'llama-3.3-70b-versatile',
    temperature = 0.6
)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data loading...Started...")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " "],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter...Started...")
    docs = text_splitter.split_documents(data)

    # create embeddings and it to FAISS index
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    # save the FAISS index to a pickle file
    with open("vectorstore.pkl", 'wb') as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query:
    with open('vectorstore.pkl', 'rb') as f:
        vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, 
            retriever=vectorstore.as_retriever()
        )
        result = chain({'question': query}, return_only_outputs=True)
        print(result)
        # {'answer': "", 'sources': ""}
        st.header("Answer")
        st.write(result['answer'])
        
        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)