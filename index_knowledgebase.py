import argparse

import bs4
import os
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredURLLoader, PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


def load_docs(documents_path):
    # Read documents
    pdf_loader = PyPDFDirectoryLoader(documents_path)
    web_loader = WebBaseLoader(
        web_paths=[
            "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendMarktakteur.html",
            "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendAnlagen.html",
            "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendFristen.html"
        ],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(["h", "article", "li"])
        )
    )
    pdf_docs = pdf_loader.load()
    web_docs = web_loader.load()
    docs = []
    docs.extend(pdf_docs)
    docs.extend(web_docs)
    return docs

def create_embeddings_from_docs(docs, save_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=save_path)



if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="takes the path to a directory containing a set of pdf-documents"
                                                    "and optionally a file containing a list of links to html docs."
                                                    "Subsequently turns those documents into embeddings and saves"
                                                    "them to a local vectordatabase.")
    argparser.add_argument("-d", "--directory",
                           help="specifies the path to the directory containing the docs.")
    argparser.add_argument("-o", "--output", help="specifies the path to where the vectordb containing"
                                                  "the embeddings is to be saved.")
    args = argparser.parse_args()

    docs = load_docs(documents_path=args.directory)
    create_embeddings_from_docs(docs, save_path=args.output)