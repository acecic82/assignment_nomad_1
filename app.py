from stmodule import getSt as st

from file import embed_file
from openaimodule import LLMManager
from openai.error import AuthenticationError


st().set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st().title("DocumentGPT")

llmManager = LLMManager()


def save_message(message, role):
    st().session_state["message"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st().chat_message(role):
        st().markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st().session_state["message"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

st().markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
"""
)

with st().sidebar:
    apiKey = st().text_input("Please set an open ai api key")
    st().session_state["api_key"] = apiKey
    print("Set the api key")
    st().write(apiKey)

    llmManager.initLLM()

    file = st().file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    st().write(
        """
        
        Github : https://github.com/acecic82/assignment_nomad_1

        app : https://assignmentnomad1-m8ps7km3kzl5lzrnhsetmu.streamlit.app/

        """
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st().chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        docs = retriever.invoke(message)
        
        try:
            chain = llmManager.getChain(retriever)
            with st().chat_message("ai"):
                chain.invoke(message)
        except AuthenticationError:
            send_message("Incorrect API key. Please check your API key", "ai")
        except Exception as e:
            print(e)
        
else:
    st().session_state["message"] = []
    st().session_state["api_key"] = ""