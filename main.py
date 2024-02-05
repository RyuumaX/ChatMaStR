import langchain_community.vectorstores.starrocks
import streamlit as st
import os
from langchain.globals import set_debug
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts.prompt import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from trulens_eval import Feedback, TruChain, feedback, Select
from langchain_core.messages import get_buffer_string
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
        print("starting on_llm_new_token")
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
    knowledgebase = load_knowledgebase(path="./KnowledgeBase/")
    embedding = HuggingFaceEmbeddings(
        model_name="T-Systems-onsite/german-roberta-sentence-transformer-v2",
        # temporarily disabled
        # model_kwargs={'device': 'cuda:1'}
    )
    # load persisted vectorstore
    vectorstore = Chroma(persist_directory="./KnowledgeBase/", embedding_function=embedding)

    parentsplitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    childsplitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
    store = InMemoryStore()
    big_chunk_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        parent_splitter=parentsplitter,
        child_splitter=childsplitter,
        search_kwargs={'k': 5}
    )
    big_chunk_retriever.add_documents(knowledgebase)

    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

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


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct. If you don't know the answer to a question, please don't share false information."""


def add_prompt_templates_together(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + system_prompt + instruction + E_INST
    return prompt_template


if __name__ == '__main__':
    set_debug(True)
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

    # Define a custom prompt for the llm to use
    sys_prompt = """Du bist ein hilfreicher, respektvoller und ehrlicher Assistent.
    Antworte immer so hilfreich wie möglich und nutze dafür den gegebenen Kontext.
    Deine Antworten sollten ausschließlich die Frage beantworten und keinen Text nach der Antwort beinhalten.
    Wenn eine Frage nicht anhand des Kontexts beantwortbar ist, sage dies und gib keine falschen Informationen.
    """
    instruction_prompt_template = """KONTEXT:/n/n {context}/n
    Frage: {question}"""

    prompt_template = add_prompt_templates_together(instruction_prompt_template, sys_prompt)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=st.session_state['temperature_slider'])
    chain_type_kwargs = {"prompt": prompt}
    retriever = configure_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=st_chat_messages, return_messages=True)
    memory_lcel = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")


    # final chain assembly
    conv_chain = ConversationalRetrievalChain.from_llm(llm,
                                                       chain_type="stuff",
                                                       combine_docs_chain_kwargs=chain_type_kwargs,
                                                       retriever=retriever,
                                                       memory=memory, )
    qa_chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", chain_type_kwargs=chain_type_kwargs, retriever=retriever, memory=memory,

    )

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # streamlit.session_state is streamlits global dictionary for savong session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="Wie kann ich helfen?")]

    for msg in st.session_state["messages"]:
        # icon = "./assets/regiocom_logo.png" if msg.role=="assistant" else ""
        # st.chat_message(msg.role, avatar=icon).write(msg.content)
        st.chat_message(msg.role).write(msg.content)

    if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
        st.session_state["messages"].append(ChatMessage(role="user", content=query))
        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            retrieval_handler = PrintRetrievalHandler(st.container())
            # finally, run the chain, which invokes the llm-chatcompletion under the hood

            response = qa_chain.run(query, callbacks=[retrieval_handler, stream_handler])
            st.session_state["messages"].append(ChatMessage(role="assistant", content=response))
