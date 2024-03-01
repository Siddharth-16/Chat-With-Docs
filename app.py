import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, user_template, bot_template

def get_pdf_text(docs):
    text = ""
    if docs is not None:
        for pdf in docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_text_embedding(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vector_store

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", layout="wide")
    
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    st.text_input("Ask a question about your documents:")

    st.write(user_template.replace(
                "{{MSG}}", "Hello human"), unsafe_allow_html=True)
    
    st.write(bot_template.replace(
                "{{MSG}}", "Hello Bot"), unsafe_allow_html=True)
    


    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                text = get_pdf_text(docs)
                # st.write(text)

                text_chunks = get_text_chunks(text)
                # st.write(text_chunks)

                text_embedding = get_text_embedding(text_chunks)

if __name__ == '__main__':
    main()
