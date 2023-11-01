import streamlit as st
import openai
import os

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, BSHTMLLoader, UnstructuredURLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

if not os.environ['OPENAI_API_BASE']:
    os.environ['OPENAI_API_BASE'] = "http://149.11.242.18:16598/v1"
#os.environ['OPENAI_API_BASE'] = "http://31.12.82.146:16700/v1"
os.environ['OPENAI_API_KEY'] = "EMPTY"
#


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
def configure_retriever():
    # Read documents
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    loaderPDF = PyPDFDirectoryLoader('./KnowledgeBase/')
    urls = [
        "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendMarktakteur.html",
        "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendAnlagen.html",
        "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendFristen.html"
    ]
    loaderURL = UnstructuredURLLoader(urls=urls)
    index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loaderPDF, loaderURL])

    # Define and configure retriever
    # retriever = index.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10})
    retriever = index.vectorstore.as_retriever()

    return retriever


if __name__ == '__main__':

    # Streamlit Configuration Stuff
    st.set_page_config(
        page_title="Lokales LLM des MaStR",
        page_icon="🤖"
    )
    st.header("Lokales LLM des MaStR")
    stream_handler = StreamHandler(st.empty())
    st_chat_messages = StreamlitChatMessageHistory()
    with st.sidebar:
        temperature_slider = st.slider(
            "Temperaturregler:",
            0.0, 1.0,
            value=0.1,
            key="temperature_slider",
        )

    # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=st.session_state['temperature_slider'])
    retriever = configure_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=st_chat_messages, return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="Wie kann ich helfen?")]

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
        st.session_state.messages.append(ChatMessage(role="user", content=query))
        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            retrieval_handler = PrintRetrievalHandler(st.container())
            response = qa_chain.run(query, callbacks=[retrieval_handler, stream_handler])
            st.session_state.messages.append(ChatMessage(role="assistant", content=response))
