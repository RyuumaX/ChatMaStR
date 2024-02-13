import streamlit as st
from langchain.globals import set_debug
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage._lc_store import create_kv_docstore
from operator import itemgetter
import bs4

from callback_handlers import StreamHandler, PrintRetrievalHandler
from prompt_templates import DEFAULT_SYSTEM_PROMPT, B_INST, E_INST, B_SYS, E_SYS, SYS_PROMPT, \
    INSTRUCTION_PROMPT_TEMPLATE, DOC_PROMPT_TEMPLATE, STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE


@st.cache_resource(ttl="1h")
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
    vectorstore = Chroma(persist_directory="./KnowledgeBase/", embedding_function=embedding)
    fs = LocalFileStore("./KnowledgeBase/store_location")
    store = create_kv_docstore(fs)
    parentsplitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    childsplitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
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


def get_pdf_docs_from_path(path):
    pdf_loader = PyPDFDirectoryLoader(path)
    pdf_docs = pdf_loader.load()
    return pdf_docs


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


if __name__ == '__main__':
    #set_debug(True)
    # Streamlit Configuration Stuff
    st.set_page_config(page_title="Lokales LLM des MaStR (Experimental)",
                       page_icon="🤖"
                       )
    st.header("Lokales LLM des MaStR (Experimental)")
    stream_handler = StreamHandler(st.empty())
    st_chat_messages = StreamlitChatMessageHistory()
    with st.sidebar:
        temperature_slider = st.slider("Temperaturregler:",
                                       0.0, 1.0,
                                       value=0.1,
                                       key="temperature_slider",
                                       )

    prompt_template = add_prompt_templates_together(INSTRUCTION_PROMPT_TEMPLATE, SYS_PROMPT)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"]
                            )

    # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=0.1)
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

    lcel_memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(lcel_memory.load_memory_variables) | itemgetter("history"),
    )
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE)
    ANSWER_PROMPT = PromptTemplate.from_template(prompt_template)
    # Now we calculate the standalone question
    make_standalone_question_chain = {
        "standalone_question": {
                                   "question": lambda x: x["question"],
                                   "chat_history": lambda x: get_buffer_string(x["chat_history"]),
                               }
                               | CONDENSE_QUESTION_PROMPT
                               | llm
                               | StrOutputParser(),
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer_chain = {
        "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
        "docs": itemgetter("docs"),
    }
    # And now we put it all together!
    lcel_qa_chain = loaded_memory | make_standalone_question_chain | retrieved_documents | answer_chain

    # streamlit.session_state is streamlits global dictionary for saving session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="Wie kann ich helfen?")]

    for msg in st.session_state["messages"]:
        # icon = "./assets/regiocom_logo.png" if msg.role=="assistant" else ""
        # st.chat_message(msg.role, avatar=icon).write(msg.content)
        st.chat_message(msg.role).write(msg.content)

    if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
        st.session_state["messages"].append(ChatMessage(role="user", content=query))
        st.chat_message("user").write(query)
        chain_input = {"question": query}

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            retrieval_handler = PrintRetrievalHandler(st.container())
            # finally, run the chain, which invokes the llm-chatcompletion under the hood

            #response = qa_chain.invoke({"query": query}, {"callbacks":[retrieval_handler, stream_handler]})
            response = qa_chain.run(query, callbacks=[retrieval_handler, stream_handler])
            print(response)
            #lcel_reponse = lcel_qa_chain.invoke(chain_input, config={"callbacks": [retrieval_handler, stream_handler]})
            if "messages" not in st.session_state:
                st.session_state["messages"] = [ChatMessage(role="assistant", content=response)]
            else:
                st.session_state["messages"].append(ChatMessage(role="assistant", content=response))
            #st.session_state["messages"].append(ChatMessage(role="assistant", content=lcel_reponse))
