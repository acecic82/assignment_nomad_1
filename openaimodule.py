from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from stmodule import getSt as st

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st().empty()

    def on_llm_end(self, *args, **kwargs):
        st().session_state["message"].append({"message": self.message, "role": "ai"})

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

class LLMManager:
    llm = ""
    
    def initLLM(self):
        print("initLLM")
        self.llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler()
            ],
            openai_api_key = st().session_state["api_key"]
        )
    
    def getChain(self, retriever):
        key = st().session_state["api_key"]
        if checkApiKey() == False:
            raise Exception("API 키가 제대로 입력되지 않았습니다.")

        return (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                } 
                | prompt 
                | self.llm
        )


def checkApiKey() -> bool:
    if len(st().session_state["api_key"]) > 0:
        return True
    return False


def createLLM():
    return ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler()
        ],
        openai_api_key=st().session_state["api_key"]
    )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

