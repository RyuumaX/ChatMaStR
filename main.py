import os.path

import bs4
import tqdm
import streamlit as st
from langchain.globals import set_debug
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import AIMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from operator import itemgetter
from callback_handlers import StreamHandler, PrintRetrievalHandler
from prompt_templates import DEFAULT_SYSTEM_PROMPT, B_INST, E_INST, B_SYS, E_SYS, SYS_PROMPT, \
    INSTRUCTION_PROMPT_TEMPLATE, DOC_PROMPT_TEMPLATE, STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE


@st.cache_resource(ttl="2h")
def configure_retriever(vectorstore_path, docstore_path="./KnowledgeBase/store_location_exp"):
    embedding = HuggingFaceEmbeddings(
        model="T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    )
    vectorstore = Chroma(
        collection_name="small_chunks",
        persist_directory=vectorstore_path,
        embedding_function=embedding
    )
    retriever = vectorstore.as_retriever()
    print(f"\n========MAIN: Vectorstore Collection Count: {vectorstore._collection.count()}=======\n")
    return retriever


def configure_parent_retriever(docstore_path, vectorstore):
    docstore = create_kv_docstore(docstore_path)
    parentsplitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200,
                                                    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
                                                    )
    childsplitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100,
                                                   separators=["\n\n", "\n", "(?<=\. )", " ", ""]
                                                   )
    big_chunk_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=parentsplitter,
        child_splitter=childsplitter,
        search_kwargs={'k': 5}
    )


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
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


if __name__ == '__main__':

    KNOWLEDGEBASE_PATH = "./KnowledgeBase/chromadb_experimental/"
    set_debug(True)
    # Streamlit Configuration Stuff
    st.header("EWI-Chatbot (Experimental)")
    st.set_page_config(page_title="EWI-Chatbot (Experimental)",
                       page_icon="🤖"
                       )
    with st.sidebar:
        temperature_slider = st.slider("Temperaturregler:",
                                       0.0, 1.0,
                                       value=0.1,
                                       key="temperature_slider",
                                       )
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
    retriever = configure_retriever(KNOWLEDGEBASE_PATH)
    chain_type_kwargs = {"prompt": prompt}
    memory = ConversationBufferWindowMemory(k=3, chat_memory=st_chat_messages, return_messages=True)
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

    #
    # lcel_memory = ConversationBufferWindowMemory(k=3, return_messages=True, output_key="answer", input_key="question")
    # loaded_memory = RunnablePassthrough.assign(
    #     chat_history=RunnableLambda(lcel_memory.load_memory_variables) | itemgetter("history"),
    # )
    # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE)
    # ANSWER_PROMPT = PromptTemplate.from_template(prompt_template)
    # # Now we calculate the standalone question
    # make_standalone_question_chain = {
    #     "standalone_question": {
    #                                "question": lambda x: x["question"],
    #                                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    #                            }
    #                            | CONDENSE_QUESTION_PROMPT
    #                            | llm
    #                            | StrOutputParser(),
    # }
    # # Now we retrieve the documents
    # retrieved_documents = {
    #     "docs": itemgetter("standalone_question") | retriever,
    #     "question": lambda x: x["standalone_question"],
    # }
    # # Now we construct the inputs for the final prompt
    # final_inputs = {
    #     "context": lambda x: combine_documents(x["docs"]),
    #     "question": itemgetter("question"),
    # }
    # # And finally, we do the part that returns the answers
    # answer_chain = {
    #     "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    #     "docs": itemgetter("docs"),
    # }
    # # And now we put it all together!
    # lcel_qa_chain = loaded_memory | make_standalone_question_chain | retrieved_documents | answer_chain

    # streamlit.session_state is streamlits global dictionary for saving session state
    # if st.session_state["message_history"]
    if len(st_chat_messages.messages) == 0:
        st_chat_messages.add_ai_message(AIMessage(content="Wie kann ich helfen?"))
    pretty(st.session_state)

    for msg in st.session_state["message_history"]:
        st.chat_message(msg.type).write(msg.content)

    if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
        if query == "killdb":
            if os.path.isfile(KNOWLEDGEBASE_PATH + "chroma.sqlite3"):
                os.remove(KNOWLEDGEBASE_PATH + "chroma.sqlite3")
        else:
            # st.session_state["message_history"].append(HumanMessage(content=query))
            st.chat_message("user").write(query)

            with st.chat_message("ai"):
                stream_handler = StreamHandler(st.empty())
                retrieval_handler = PrintRetrievalHandler(st.container())
                # finally, run the chain, which invokes the llm-chatcompletion under the hood
                # response = qa_chain.invoke({"query": query},
                #                            {"callbacks": [retrieval_handler,stream_handler]})
                response = conv_chain.invoke({"question": query},
                                             {"callbacks": [retrieval_handler, stream_handler]})

                print("=====RESPONSE=====")
                pretty(response, indent=2)
                # st.session_state["message_history"].append(AIMessage(content=response["result"]))
                print("=====STREAMLIT SESSION DICT=====")
                pretty(st.session_state, indent=2)
