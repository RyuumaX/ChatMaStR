import streamlit as st
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bs4


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Relevante Dokumente**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Anfrage:** {query}")
        self.status.update(label=f"**Relevante Dokumente:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Dokument {idx}: {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


@st.cache_resource(ttl="1h")
def configure_retriever(knowledgebase):

    embedding = HuggingFaceEmbeddings(
        model_name="T-Systems-onsite/german-roberta-sentence-transformer-v2",
        #temporarily disabled
        #model_kwargs={'device': 'cuda:1'}
    )
    # load persisted vectorstore
    vectorstore = Chroma(persist_directory="./KnowledgeBase/", embedding_function=embedding)

    parentsplitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    childsplitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    store = InMemoryStore()
    big_chunk_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        store=store,
        parentsplitter=parentsplitter,
        childsplitter=childsplitter
    )
    big_chunk_retriever.add_documents(knowledgebase)

    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    return big_chunk_retriever


def load_knowledgebase(path):
    # Read documents
    pdf_loader = PyPDFDirectoryLoader(path)
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


if __name__ == '__main__':
    # Streamlit Configuration Stuff
    st.set_page_config(
        page_title="Lokales LLM des MaStR (Experimental)",
        page_icon="🤖"
    )
    st.header("Lokales LLM des MaStR (Experimental)")
    stream_handler = StreamHandler(st.empty())
    st_chat_messages = StreamlitChatMessageHistory()

    #RAG Retrieval Step - Langchain Version
    # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=0)
    knowledgebase = load_knowledgebase(path="./KnowledgeBase/")
    retriever = configure_retriever(knowledgebase)
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=st_chat_messages, return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, chain_type="stuff", retriever=retriever, memory=memory,
    )


    #streamlit.session_state is streamlits global dictionary for savong session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="Wie kann ich helfen?")]

    for msg in st.session_state.messages:
        #icon = "./assets/regiocom_logo.png" if msg.role=="assistant" else ""
        #st.chat_message(msg.role, avatar=icon).write(msg.content)
        st.chat_message(msg.role).write(msg.content)

    if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
        st.session_state.messages.append(ChatMessage(role="user", content=query))
        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            retrieval_handler = PrintRetrievalHandler(st.container())
            #finally, run the chain, which invokes the llm-chatcompletion under the hood
            response = qa_chain.run(query, callbacks=[retrieval_handler, stream_handler])
            st.session_state.messages.append(ChatMessage(role="assistant", content=response))
