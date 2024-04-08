from operator import itemgetter

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.memory import BaseMemory
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain.globals import set_debug
from callback_handlers import StreamHandler, PrintRetrievalHandler
from prompt_templates import DEFAULT_SYSTEM_PROMPT, B_INST, E_INST, B_SYS, E_SYS, SYS_PROMPT, \
    INSTRUCTION_PROMPT_TEMPLATE, DOC_PROMPT_TEMPLATE, STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE


@st.cache_resource(ttl="2h")
def configure_retriever(vectorstore_path):
    embedding = HuggingFaceEmbeddings(model_name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer")
    vectorstore = Chroma(collection_name="small_chunks",
                         persist_directory=vectorstore_path,
                         embedding_function=embedding
                         )
    retriever = vectorstore.as_retriever(search_type="similarity")
    print(f"\n========Vectorstore Collection Count: {vectorstore._collection.count()}=======\n")
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
    set_debug(True)
    KNOWLEDGEBASE_PATH = "./KnowledgeBase/chromadb_experimental/"
    # Streamlit Configuration Stuff
    st.set_page_config(page_title="EWI-Chatbot (Experimental)",
                       page_icon="🤖"
                       )
    st.header("EWI-Chatbot (Experimental)")
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
    chat_history = StreamlitChatMessageHistory(key="message_history")
    # Here ends Streamlit configuration

    # Here begins the actual langchain code.
    # prompt_template = add_prompt_templates_together(INSTRUCTION_PROMPT_TEMPLATE, SYS_PROMPT)
    # query = PromptTemplate(template=prompt_template,
    #                        input_variables=["context", "question"]
    #                        )
    # condense_question_prompt = PromptTemplate.from_template(STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE)
    # answer_prompt = PromptTemplate.from_template(prompt_template)
    #
    # # LLM configuration. ChatOpenAI is merely a config object
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=st.session_state["temperature_slider"])
    # retriever = configure_retriever(KNOWLEDGEBASE_PATH)
    # chain_type_kwargs = {"prompt": query}
    #
    # memory = ConversationBufferWindowMemory(k=3, chat_memory=chat_history, return_messages=True)
    # loaded_memory = RunnablePassthrough.assign(
    #     chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    # )
    # # Now we calculate the standalone question
    # make_standalone_question_chain = {
    #     "standalone_question": {
    #                                "question": lambda x: x["question"],
    #                                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    #                            }
    #                            | condense_question_prompt
    #                            | llm
    #                            | StrOutputParser(),
    # }
    # # Now we retrieve the documents
    # retrieved_documents = {
    #     "docs": itemgetter("standalone_question") | retriever,
    #     "question": itemgetter("standalone_question"),
    # }
    # # Now we construct the inputs for the final prompt
    # final_inputs = {
    #     "context": lambda x: combine_documents(x["docs"]),
    #     "question": itemgetter("question"),
    # }
    # # And finally, we do the part that returns the answers
    # answer_chain = (
    #     {
    #         "answer": final_inputs | answer_prompt | ChatOpenAI(),
    #         "docs": itemgetter("docs"),
    #     }
    # )
    # # And now we put it all together!
    # lcel_chain_with_history = loaded_memory | make_standalone_question_chain | retrieved_documents | answer_chain
    #
    # # final chain assembly
    # conv_chain = ConversationalRetrievalChain.from_llm(llm,
    #                                                    chain_type="stuff",
    #                                                    combine_docs_chain_kwargs=chain_type_kwargs,
    #                                                    retriever=retriever,
    #                                                    memory=ConversationBufferWindowMemory(k=3),
    #                                                    verbose=True
    #                                                    )

    # streamlit.session_state is streamlits global dictionary for saving session state
    # if st.session_state["message_history"]
    if len(chat_history.messages) == 0:
        chat_history.add_ai_message("Wie kann ich helfen?")

    # query = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are an AI chatbot having a conversation with a human."),
    #         MessagesPlaceholder(variable_name="history"),
    #         ("human", "{question}"),
    #     ]
    # )

    template = """You are an AI Chatbot having a conversation with a human:
    {history}
    
    Answer the humans questions based on the given context:
    Kontext: {context}

    Question: {question}
    """
    query = ChatPromptTemplate.from_template(template)

    chain = query | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,  # Always return the instance created earlier
        input_messages_key="question",
        history_messages_key="history",
    )

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if query := st.chat_input():
        st.chat_message("human").write(query)

        # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
        config = {"configurable": {"session_id": "any"}}
        response = chain_with_history.invoke({"question": query, "context": "Carl Jonson is a member of the groove street gang."}, config)
        print(chat_history.messages)
        st.chat_message("ai").write(response.content)

    # pretty(st.session_state)
    # print(f"\n=============BISHERIGER CHAT VERLAUF===================\n {memory.buffer}\n")
    #
    # for msg in st.session_state["message_history"]:
    #     st.chat_message(msg.type).write(msg.content)
    #
    # if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
    #     # st.session_state["message_history"].append(HumanMessage(content=query))
    #     st.chat_message("user").write(query)
    #
    #     with st.chat_message("ai"):
    #         stream_handler = StreamHandler(st.empty())
    #         retrieval_handler = PrintRetrievalHandler(st.container())
    #
    #         # finally, run the chain, which invokes the llm-chatcompletion under the hood
    #         response = conv_chain.invoke({"question": query, "chat_history": chat_history.messages},
    #                                      {"callbacks": [retrieval_handler, stream_handler]})
    #
    #         # retrieval_response = chain_with_history.invoke({"question": prompt}, config)
    #         # print(retrieval_response)
    #
    #         # lcel_response = lcel_chain_with_history.invoke({"question": query, "chat_history": memory},
    #         #                                              {"callbacks": [retrieval_handler, stream_handler]})
    #
    #         print(f"\n=====RESPONSE=====\n")
    #         # pretty(lcel_response, indent=2)
    #         # st.session_state["message_history"].append(AIMessage(content=response["result"]))
    #         print(f"\n=====STREAMLIT SESSION DICT=====\n")
    #         pretty(st.session_state, indent=2)
