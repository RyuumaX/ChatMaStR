import argparse
from os import listdir
from os.path import isfile, join

import bs4
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import (WebBaseLoader, PyPDFLoader)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm


def get_pdf_docs_from_path(path):
    documents = []
    num_pdfs = 0
    for f in tqdm(listdir(path)):
        if isfile(join(path, f)) and f.endswith(".pdf"):
            num_pdfs += 1
            documents.extend(PyPDFLoader(join(path, f)).load())
    print(f"Anzahl PDF-docs: {num_pdfs}")
    return documents


def get_web_docs_from_urls(path):
    fullpath = join(path, "webdocs.txt")
    if isfile(fullpath):
        urls = [url for url in fullpath]
        web_loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(["h", "article", "li"])
            )
        )
        web_docs = web_loader.load()
    else:
        web_docs = []
    print(f"Anzahl Web-Docs: {len(web_docs)}")
    return web_docs


def load_docs_from_path(path):
    # Read documents
    pdf_docs = get_pdf_docs_from_path(path)
    print(f"Anzahl Splits von PDF-docs: {len(pdf_docs)}")
    web_docs = get_web_docs_from_urls(path)
    print(f"Anzahl Splits von Webdocs: {len(web_docs)}")
    all_docs = []
    all_docs.extend(pdf_docs)
    all_docs.extend(web_docs)
    print(f"Insgesamt {str(len(all_docs))} Chunks/Splits.")
    return all_docs


def create_vectordb_for_texts(texts, save_path):
    embedding_model = HuggingFaceEmbeddings(
        model_name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    )
    test_embedding = embedding_model.embed_documents(texts[0].page_content)
    n = 5
    print(f"\n==========FIRST EMBEDDING ({len(test_embedding[:n])} OF {len(test_embedding[0])} DIMENSIONS:=========\n")
    print(test_embedding[0][:n], "...")
    vectorstore = Chroma.from_documents(collection_name="small_chunks", documents=texts, embedding=embedding_model,
                                        persist_directory=save_path)
    return vectorstore


def split_texts_into_chunks(texts, chunksize, chunk_overlap=0, splitter="recursive"):
    if splitter == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunk_overlap,
                                                  separators=["\n\n", "\n", "(?<=\. )", " ", ""]
                                                  )
    elif splitter == "character":
        splitter = CharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunk_overlap,
                                         separators=["\n\n", "\n", "(?<=\. )", " ", ""]
                                         )
    else:
        splitter = TokenTextSplitter(chunk_size=chunksize, chunk_overlap=chunk_overlap,
                                     separators=["\n\n", "\n", "(?<=\. )", " ", ""]
                                     )
    splits = splitter.split_documents(texts)
    return splits


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="takes the path to a directory containing a set of pdf-documents"
                                                    "and optionally a file containing a list of links to html docs."
                                                    "Subsequently turns those documents into embeddings and saves"
                                                    "them to a local vectordatabase.")
    argparser.add_argument("-d", "--directory",
                           help="specifies the path to the directory containing the docs.")
    argparser.add_argument("-o", "--output", help="specifies the path to where the vectordb containing"
                                                  "the embeddings is to be saved.")
    argparser.add_argument("--splittype", help="specifies the type of textsplitter to use for"
                                               "textsplitting/chunking.", default="recursive")
    argparser.add_argument("--overlap", help="specifies the amount of overlap between text splits/chunks",
                           default=0, type=int)
    argparser.add_argument("--splitsize", help="specifies the size of the splits. Actual size will"
                                               "depend on the type of splitter used (characters or tokens).",
                           default=1000, type=int)

    args = argparser.parse_args()
    splitter = args.splittype
    overlap = args.overlap
    splitsize = args.splitsize

    # Load documents for from given path. These docs become the knowledgebase for the RAG-system
    docs = load_docs_from_path(path=args.directory)

    # Split the full documents into smaller chunks (sub-documents)
    chunks = split_texts_into_chunks(docs, chunksize=splitsize, chunk_overlap=overlap, splitter=splitter)
    print(f"\n==========FIRST 3 OF {len(chunks)} SPLITS==========\n")
    for chunk in chunks[:3]:
        print(chunk, "\n")

    # create a vector database containing the embeddings for the given texts (usually document chunks/spits)
    vectordb = create_vectordb_for_texts(chunks, save_path=args.output)
    print(f"Embeddings in collection: {vectordb._collection.count()}")
    vectordb.persist()
