import base64
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
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from st_clickable_images import clickable_images


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
    system_prompt = new_system_prompt
    prompt_template = system_prompt + instruction
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

def show_big_img(img):
    st.session_state["big_img"] = img


if __name__ == '__main__':

    # Streamlit UI/Page Configuration Stuff
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
    # stream_handler = StreamHandler(st.empty())
    # Here ends Streamlit UI configuration ============================================================================

    set_debug(True)
    KNOWLEDGEBASE_PATH = "./KnowledgeBase/chromadb_experimental/"
    # # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=st.session_state["temperature_slider"])
    retriever = configure_retriever(KNOWLEDGEBASE_PATH)
    # StreamlitChatMessageHistory() handles adding Messages (AI, Human etc.) to the streamlit session state dictionary.
    # So there is no need to handle that on our own, e.g. no need to do something like
    # "st.session_state["messages"].append(msg)".
    chat_history = StreamlitChatMessageHistory(key="message_history")
    if len(chat_history.messages) == 0:
        chat_history.add_ai_message("Wie kann ich helfen?")

    # Here begins the chain build-up
    prompt_template = add_prompt_templates_together(INSTRUCTION_PROMPT_TEMPLATE, SYS_PROMPT)
    prompt = PromptTemplate.from_template(prompt_template)
    make_standalone_query_prompt = PromptTemplate.from_template(STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE)

    memory = ConversationBufferWindowMemory(k=3, chat_memory=chat_history, return_messages=True)
    loaded_memory = RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables)
                                                            | itemgetter("history")
                                               )
    # Calculate the standalone question
    make_standalone_query_chain = {
        "standalone_question": {
                                   "question": lambda x: x["question"],
                                   "history": lambda x: get_buffer_string(x["history"]),
                               }
                               | make_standalone_query_prompt
                               | llm
                               | StrOutputParser(),
        "history": lambda x: get_buffer_string(x["history"])
    }
    # Retrieve the documents
    retrieve_documents_chain = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": itemgetter("standalone_question"),
        "history": itemgetter("history")
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: combine_documents(x["docs"]),
        "question": itemgetter("question"),
        "history": itemgetter("history")
    }
    # And finally, we do the part that returns the answers
    answer_chain = final_inputs | prompt | llm

    retrieve_documents_with_history_chain = loaded_memory | make_standalone_query_chain | retrieve_documents_chain

    chain_with_history = retrieve_documents_with_history_chain | answer_chain

    # template = """You are an AI Chatbot having a conversation with a human:
    # {history}
    #
    # Answer the humans questions based on the given context:
    # Kontext: {context}
    #
    # If you cannot answer a question based on the given context, say that you cannot answer the question with the
    # context provided.
    #
    # Question: {question}
    # """
    # prompt = ChatPromptTemplate.from_template(template)
    #
    # chain = (
    #         {
    #             "context": itemgetter("question") | retriever,
    #             "question": itemgetter("question"),
    #             "history": itemgetter("history")
    #         }
    #         | prompt
    #         | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # )
    # chain_with_history = RunnableWithMessageHistory(
    #     chain,
    #     lambda session_id: chat_history,  # Always return the instance created earlier
    #     input_messages_key="question",
    #     history_messages_key="history",
    # )

    # write out all messages to the streamlit page that are already in the chat history.
    for idx, msg in enumerate(chat_history.messages):
        with st.chat_message(msg.type):
            st.write(msg.content)
            if msg.type == "ai" and msg.content != "Wie kann ich helfen?":
                with st.expander("Bilderstrecke"):

                    paths = ["./images/test.jpg", "./images/Kowalski_analyse2.jpg"]
                    # images könnte eine liste von Bildern im st_session_state dict werden die zur jeweiligen Antwort des
                    # LLMs gehört
                    images = []
                    for file in paths:
                        with open(file, "rb") as image:
                            encoded = base64.b64encode(image.read()).decode()
                            images.append(f"data:image/jpeg;base64,{encoded}")

                    clicked = clickable_images(
                        images,
                        titles=[f"Image #{str(i)}" for i in range(len(images))],
                        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                        img_style={"margin": "5px", "height": "100px"},
                        key=f"image_gallery_{idx}"
                    )

                    placeholder = st.container(height=500)
                    with placeholder.container():
                        if clicked:
                            placeholder.image(images[clicked])
                        else:
                            placeholder.image(images[0])

    # give the user an input field and write out his query/message once he submits it
    if query := st.chat_input():
        st.chat_message("human").write(query)
        with st.chat_message("ai"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            # New messages are added to StreamlitChatMessageHistory when the Chain is called.
            config = {"configurable": {"session_id": "any"},
                      "callbacks": [retrieval_handler]}
            # print(retrieve_documents_with_history_chain.invoke({"question": query}), config)
            response = st.write_stream(chain_with_history.stream({"question": query}, config))
            chat_history.add_user_message(query)
            chat_history.add_ai_message(response)
            with st.expander("Bilderstrecke"):
                paths = ["./images/test.jpg", "./images/Kowalski_analyse2.jpg"]
                # images könnte eine liste von Bildern im st_session_state dict werden die zur jeweiligen Antwort des
                # LLMs gehört
                images = []
                for file in paths:
                    with open(file, "rb") as image:
                        encoded = base64.b64encode(image.read()).decode()
                        images.append(f"data:image/jpeg;base64,{encoded}")

                clicked = clickable_images(
                    images,
                    titles=[f"Image #{str(i)}" for i in range(len(images))],
                    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                    img_style={"margin": "5px", "height": "100px"},
                    key="image_gallery"
                )

                placeholder = st.container(height=500)
                with placeholder.container():
                    if clicked:
                        placeholder.image(images[clicked])
                    else:
                        placeholder.image(images[0])

    # pretty(st.session_state)
    # print(f"\n=============BISHERIGER CHAT VERLAUF===================\n {chat_history.buffer}\n")
    # write out all messages to the streamlit page that are already in the chat history.
    # for msg in chat_history.messages:
    #     st.chat_message(msg.type).write(msg.content)
    #
    # if query := st.chat_input('Geben Sie hier Ihre Anfrage ein.'):
    #     st.chat_message("user").write(query)
    #
    #     with st.chat_message("ai"):
    #         stream_handler = StreamHandler(st.empty())
    #         retrieval_handler = PrintRetrievalHandler(st.container())
    #         # response = lcel_chain_with_history.invoke({"question": query, "chat_history": history},
    #         #                                              {"callbacks": [retrieval_handler, stream_handler]})
    #
    #         print(f"\n=====RESPONSE=====\n")
    #         # pretty(lcel_response, indent=2)
    #         # st.session_state["message_history"].append(AIMessage(content=response["result"]))
    #         print(f"\n=====STREAMLIT SESSION DICT=====\n")
    #         pretty(st.session_state, indent=2)
