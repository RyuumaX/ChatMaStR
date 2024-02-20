import bs4
import streamlit as st
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import ChatMessage
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import format_document
from langchain_openai.chat_models import ChatOpenAI
import json

from callback_handlers import StreamHandler, PrintRetrievalHandler
from prompt_templates import DEFAULT_SYSTEM_PROMPT, B_INST, E_INST, B_SYS, E_SYS, SYS_PROMPT, \
    INSTRUCTION_PROMPT_TEMPLATE, DOC_PROMPT_TEMPLATE


@st.cache_resource(ttl="4h")
def configure_retriever():
    knowledgebase = get_pdf_docs_from_path(path="./KnowledgeBase/")
    knowledgebase.extend(get_web_docs_from_urls([
        "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendMarktakteur.html",
        "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendAnlagen.html",
        "https://www.marktstammdatenregister.de/MaStRHilfe/subpages/registrierungVerpflichtendFristen.html"
    ]))

    embedding = HuggingFaceEmbeddings(
        model_name="T-Systems-onsite/german-roberta-sentence-transformer-v2",
        # temporarily disabled
        # model_kwargs={'device': 'cuda:1'}
    )
    # load persisted vectorstore
    vectorstore = Chroma(collection_name="small_chunks", persist_directory="./KnowledgeBase/chromadb_prod", embedding_function=embedding)
    fs = LocalFileStore("./KnowledgeBase/store_location")
    store = create_kv_docstore(fs)
    parentsplitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    childsplitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    #store = InMemoryStore()
    big_chunk_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        parent_splitter=parentsplitter,
        child_splitter=childsplitter,
        search_kwargs={'k': 5}
    )
    big_chunk_retriever.add_documents(knowledgebase)
    return big_chunk_retriever

@st.cache_data
def get_pdf_docs_from_path(path):
    pdf_loader = PyPDFDirectoryLoader(path)
    pdf_docs = pdf_loader.load()
    return pdf_docs


@st.cache_data
def get_web_docs_from_urls(urls):
    web_loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(["h", "article", "li"])
        )
    )
    web_docs = web_loader.load()
    return web_docs


def add_prompt_templates_together(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + system_prompt + instruction + E_INST
    return prompt_template


def combine_documents(docs,
                      document_prompt=PromptTemplate.from_template(DOC_PROMPT_TEMPLATE),
                      document_separator="\n\n"
                      ):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


if __name__ == '__main__':
    #set_debug(True)
    # Streamlit Configuration Stuff
    st.set_page_config(
        page_title="MaStR Chat-Assistent",
        page_icon="./resources/regiocom.png",
        layout="wide"
    )

    with st.sidebar:
        temperature_slider = st.slider("Temperaturregler:",
                                       0.0, 1.0,
                                       value=0.1,
                                       key="temperature_slider",
                                       )

    _, col1, _ = st.columns([0.5, 4, 0.5])
    with col1:
        st.image("./resources/logo_banner_no_bnetza.png", use_column_width=True)

    _, col2, _ = st.columns([2, 1, 2])
    with col2:
        st.header("MaStR Chat-Assistent")

    stream_handler = StreamHandler(st.empty())
    # StreamlitChatMessageHistory() handles adding Messages (AI, Human etc.) to the streamlit session state dictionary.
    # So there is no need to handle that on our own, e.g. no need to do something like
    # st.session_state["messages"].append(msg).
    st_chat_messages = StreamlitChatMessageHistory(key="message_history")
    print(st_chat_messages)

    prompt_template = add_prompt_templates_together(INSTRUCTION_PROMPT_TEMPLATE, SYS_PROMPT)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"]
                            )

    # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=st.session_state["temperature_slider"])
    retriever = configure_retriever()
    chain_type_kwargs = {"prompt": prompt}
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=st_chat_messages, return_messages=True)

    # final chain assembly
    conv_chain = ConversationalRetrievalChain.from_llm(llm,
                                                       chain_type="stuff",
                                                       combine_docs_chain_kwargs=chain_type_kwargs,
                                                       retriever=retriever,
                                                       memory=memory
                                                       )
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           chain_type="stuff",
                                           chain_type_kwargs=chain_type_kwargs,
                                           retriever=retriever,
                                           memory=memory
                                           )


    # streamlit.session_state is streamlits global dictionary for saving session state
    #if st.session_state["message_history"]
    if len(st_chat_messages.messages) == 0:
        st_chat_messages.add_ai_message(AIMessage(content="Wie kann ich helfen?"))
    pretty(st.session_state)

    for msg in st.session_state["message_history"]:
        st.chat_message(msg.type).write(msg.content)

    if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
        #st.session_state["message_history"].append(HumanMessage(content=query))
        st.chat_message("user").write(query)

        with st.chat_message("ai"):
            stream_handler = StreamHandler(st.empty())
            retrieval_handler = PrintRetrievalHandler(st.container())
            # finally, run the chain, which invokes the llm-chatcompletion under the hood

            response = qa_chain.invoke({"query": query}, {"callbacks": [retrieval_handler, stream_handler]})
            #response = conv_chain.invoke({"question": query}, {"callbacks": [retrieval_handler, stream_handler]})
            #response = qa_chain.run(query, callbacks=[retrieval_handler, stream_handler])
            print("=====RESPONSE=====")
            pretty(response, indent=2)
            #st.session_state["message_history"].append(AIMessage(content=response["result"]))
            print("=====STREAMLIT SESSION DICT=====")
            pretty(st.session_state, indent=2)
