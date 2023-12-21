import streamlit as st
import openai
import os

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import initialize_agent
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage


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
    retriever = index.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.8})
    #retriever = index.vectorstore.as_retriever()

    return retriever


def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
    ):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index

def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
    ):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


if __name__ == '__main__':

    # Streamlit Configuration Stuff
    st.set_page_config(
        page_title="Lokales LLM des MaStR (Experimental)",
        page_icon="🤖"
    )
    st.header("Lokales LLM des MaStR (Experimental)")
    stream_handler = StreamHandler(st.empty())
    st_chat_messages = StreamlitChatMessageHistory()
    with st.sidebar:
        temperature_slider = st.slider(
            "Temperaturregler:",
            0.0, 1.0,
            value=0.1,
            key="temperature_slider",
        )

    #RAG Retrieval Step - Langchain Version
    # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=st.session_state['temperature_slider'])
    retriever = configure_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=st_chat_messages, return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory
    )

    #RAG Retrieval Step - LlamaIndex Version
    


    #streamlit.session_state is streamlits global dictionary for savong session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="Wie kann ich helfen?")]

    for msg in st.session_state.messages:
        icon = "./assets/regiocom_logo.png" if msg.role=="assistant" else ""
        st.chat_message(msg.role, avatar=icon).write(msg.content)

    if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
        st.session_state.messages.append(ChatMessage(role="user", content=query))
        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            retrieval_handler = PrintRetrievalHandler(st.container())
            #finally, run the chain, which invokes the llm-chatcompletion under the hood
            response = qa_chain.run(query, callbacks=[retrieval_handler, stream_handler])
            st.session_state.messages.append(ChatMessage(role="assistant", content=response))
