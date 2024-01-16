import streamlit as st
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

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
    embedding = HuggingFaceEmbeddings(
        model_name="T-Systems-onsite/german-roberta-sentence-transformer-v2"
    )

    # load persisted vectorstore
    vectorstore = Chroma(persist_directory="./KnowledgeBase/", embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    return retriever


if __name__ == '__main__':

    # Streamlit Configuration Stuff
    st.set_page_config(
        page_title="MaStR Chat-Assistent",
        page_icon="./resources/regiocom.png",
        layout="wide"
    )
    
    col1, col2, col3 = st.columns([0.5, 4, 0.5])
    with col1:
        st.write(' ')

    with col2:
        st.image("./resources/logo_banner_no_bnetza.png", use_column_width=True)
        st.header("MaStR Chat-Assistent")

    with col3:
        st.write(' ')

    stream_handler = StreamHandler(st.empty())
    st_chat_messages = StreamlitChatMessageHistory()

    # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=0.1)
    retriever = configure_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=st_chat_messages, return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory
    )

    # streamlit.session_state is streamlits global dictionary for savong session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="Wie kann ich helfen?")]
        #streamlit.session_state is streamlits global dictionary for savong session state
        
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
        st.session_state.messages.append(ChatMessage(role="user", content=query))
        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            retrieval_handler = PrintRetrievalHandler(st.container())
            # finally, run the chain, which invokes the llm-chatcompletion under the hood
            response = qa_chain.invoke(query, callbacks=[retrieval_handler, stream_handler])
            st.session_state.messages.append(ChatMessage(role="assistant", content=response))