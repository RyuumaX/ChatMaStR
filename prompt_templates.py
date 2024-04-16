B_INST = "[INST]"
E_INST = "[/INST]"
B_SYS = "<<SYS>>\n"
E_SYS = "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct. If you don't know the answer to a question, please don't share false information."""


SYS_PROMPT = """Du bist ein hilfreicher, respektvoller und ehrlicher Assistent.
Antworte immer so hilfreich wie möglich und nutze dafür den gegebenen Kontext.
Deine Antworten sollten ausschließlich die Frage beantworten und keinen Text nach der Antwort beinhalten.
Wenn du eine Frage nicht anhand des Kontexts beantworten kannst, sage dass du die Frage anhand des gegebenen
Kontextes nicht beantworten kannst.
"""


INSTRUCTION_PROMPT_TEMPLATE = """KONTEXT:\n\n {context}\n\n
Chat_Historie: {history}\n\n
Frage: {question}"""


DOC_PROMPT_TEMPLATE = "{page_content}"


STANDALONE_QUESTION_FROM_HISTORY_TEMPLATE = """
Gegeben ist die unten stehende Konversation zwischen einem Menschen und dem AI-Assistenten, sowie eine Folgefrage.
Formuliere die Folgefrage in eine eigenständige Frage um und beachte dabei die vorherige Konversation.

Konversation:
{chat_history}
Folgefrage: {question}
Eigenständige Frage:
"""


FINAL_PROMPT_TEMPLATE = """
Das Folgende ist eine freundliche Unterhaltung zwischen einem Menschen und einem AI-Assistenten. Beantworte die Folgefrage,
basierend auf dem gegebenen Kontext und dem bisherigen Gesprächsverlauf.\n Wenn die Frage nicht aus dem gegebenen Kontext
zu beantworten ist, sage, dass du anhand der gegebenen Dokumente keine Antwort geben kannst.\n
Kontext: {context}\n
Chat_Historie: {chat_history}\n
Folgefrage: {standalone_question}
"""