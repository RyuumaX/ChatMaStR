import base64
import os
from operator import itemgetter
from os import path

import streamlit as st
from langchain.globals import set_debug
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from st_clickable_images import clickable_images

from prompt_templates import DEFAULT_SYSTEM_PROMPT, SYS_PROMPT, \
    INSTRUCTION_PROMPT_TEMPLATE, DOC_PROMPT_TEMPLATE, STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE


@st.cache_resource(ttl="2h")
def configure_retriever(vectorstore_path):
    """
    Method that takes a path to a vectorestore and returns a retriever object to interact with that vectorstore.
    :param vectorstore_path: path to the actual vectorstore file (sqlite file in our case)
    :return: retriever object that acts as a wrapper around the vector database
    """
    if path.exists(vectorstore_path):
        print(f"Found Knowledgebase at {vectorstore_path}")
    else:
        print("No Knowledgebase found!")
    embedding = HuggingFaceEmbeddings(model_name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer")
    vectorstore = Chroma(collection_name="small_chunks",
                         persist_directory=vectorstore_path,
                         embedding_function=embedding
                         )
    # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
    #                                    search_kwargs={"k": 6, "score_threshold": 0.64})

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print(f"\n========Vectorstore Collection Count: {vectorstore._collection.count()}=======\n")
    return retriever


def add_prompt_templates_together(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    """
    Joins several prompt templates together. Usually an Instruction Prompt and a system prompt.
    :param instruction: Abitrary Instruction Prompt.
    :param new_system_prompt: System Prompt to be used.
    :return: joined prompt
    """
    system_prompt = new_system_prompt
    prompt_template = system_prompt + "\n" + instruction
    return prompt_template


def combine_documents(docs,
                      document_prompt=PromptTemplate.from_template(DOC_PROMPT_TEMPLATE),
                      document_separator="\n\n"
                      ):
    """
    Combines several documents into one string for later use with an LLM
    :param docs: List of document objects to be combined.
    :param document_prompt: Prompt that extracts the page content of a document object.
    :param document_separator: seperator to put in between the documents page contents when creating the resulting
    string
    :return: String containing all documents' page contents
    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def pretty(d, indent=0):
    """
    Simple helper method for pretty printing the contents of a dictionary with indentation.
    :param d: the dictionary to be printed
    :param indent: indentation depth
    :return:
    """
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def trim_question(text: str) -> str:
    """
    Takes a string and trimms off all given sub strings.
    :param text: Text to be trimmed.
    :return: returns new string without all the trimmed off sub strings.
    """

    substrings = ["Standalone-Folgefrage:", "Standalone question:", "Standalone Frage:"]
    for substring in substrings:
        trimmed_text = text.replace(substring, "").strip()
        text = trimmed_text

    return text


if __name__ == '__main__':

    # Streamlit UI/Page Configuration Stuff
    pagetitle = os.environ["ST_PAGETITLE"]
    st.set_page_config(page_title=pagetitle,
                       page_icon="🤖"
                       )
    st.header(pagetitle)
    with st.sidebar:
        temperature_slider = st.slider("Temperaturregler:",
                                       0.0, 1.0,
                                       value=0.1,
                                       key="temperature_slider",
                                       )
    # Here ends Streamlit UI configuration ============================================================================

    set_debug(True)

    KNOWLEDGEBASE_PATH = "./KnowledgeBase/chromadb_experimental/"
    IMAGE_PATH = "./KnowledgeBase/images"
    STOPWORD_PATH = "./KnowledgeBase/stopwords_german.txt"

    if path.exists(STOPWORD_PATH):
        stopwords = []
        with open(STOPWORD_PATH) as file:
            for line in file:
                line.strip()
                stopwords.extend(line.split(", "))

    # LLM configuration. ChatOpenAI is merely a config object
    llm = ChatOpenAI(streaming=True, temperature=st.session_state["temperature_slider"])
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

    memory = ConversationBufferWindowMemory(k=5, chat_memory=chat_history, return_messages=True)
    loaded_memory = RunnablePassthrough.assign(history=
                                               RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
    # Calculate the standalone question
    make_standalone_query_chain = {
        "standalone_question": {
                                   "question": lambda x: x["question"],
                                   "history": lambda x: get_buffer_string(x["history"]),
                               }
                               | make_standalone_query_prompt
                               | llm
                               | StrOutputParser()
                               | RunnableLambda(trim_question),
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

    # write out all messages to the streamlit page that are already in the chat history.
    for idx, msg in enumerate(chat_history.messages):
        with st.chat_message(msg.type):
            st.write(msg.content)
            if msg.type == "ai" and msg.content != "Wie kann ich helfen?" and os.environ["ST_SHOW_GALLERY"] == "TRUE":
                with st.expander("Bilderstrecke"):

                    paths = [f"{IMAGE_PATH}/test1.png", f"{IMAGE_PATH}/test2.png"]
                    # images could become a list of images in the st_session_state dict that belong to the respective
                    # LLMs answer
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

    # give the user an input field and write out their query/message once they submit it
    if query := st.chat_input():
        # search_query = remove_words_from_text(query, stopwords)
        st.chat_message("human").write(query)
        with st.chat_message("ai"):
            # New messages are added to StreamlitChatMessageHistory when the Chain is called.
            config = {"configurable": {"session_id": "any"}}
            # print(retrieve_documents_with_history_chain.invoke({"question": query}), config)
            # calling stream is the same as calling invoke for a chain but returns an iterator so we can display the
            # LLMs answer as it is being generated
            response = st.write_stream(chain_with_history.stream({"question": query}, config))
            chat_history.add_user_message(query)
            chat_history.add_ai_message(response)

            with st.expander("Zurückgelieferte Dokumente"):
                retrieved_docs = retriever.invoke(query)
                for doc in retrieved_docs:
                    source = path.basename(doc.metadata["source"])
                    st.markdown("**" + source + "**")
                    st.markdown(doc.page_content)
            if os.environ["ST_SHOW_GALLERY"] == "TRUE":
                with st.expander("Bilderstrecke"):
                    paths = [f"{IMAGE_PATH}/test1.png", f"{IMAGE_PATH}/test2.png"]
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
