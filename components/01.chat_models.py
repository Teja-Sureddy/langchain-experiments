"""
Interact with chat-based language models in a conversational format.
"""
import textwrap
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

# -------------------- Ollama --------------------
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class ChatSession:
    def __init__(self, model: str = 'llama3.2', instruction: str = 'Make the response short') -> None:
        self.llm = ChatOllama(model=model)
        self.history = FileChatMessageHistory(file_path='../tmp/history.json')
        if not self.history.messages or not isinstance(self.history.messages[0], SystemMessage):
            self.history.add_message(SystemMessage(content=instruction))

        self.memory = ConversationBufferMemory(chat_memory=self.history, return_messages=True)
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=False)

    def start(self):
        while True:
            prompt = input("\nEnter prompt: ") or "Hello, my name is teja"
            response = self.conversation.predict(input=prompt)
            print(textwrap.fill(response, width=100))


# -------------------- Multi-Modal --------------------
import base64
from langchain_community.chat_message_histories import ChatMessageHistory


def multi_modal(model: str = 'llava', instruction: str = 'Make the response short'):
    """
    Supports text and image input.
    """
    llm = ChatOllama(model=model, temperature=0)
    history = ChatMessageHistory()

    img_b64 = base64.b64encode(open('../assets/picture.jpg', 'rb').read()).decode()
    history.add_messages([
        SystemMessage(instruction),
        HumanMessage([{"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}])
    ])

    while True:
        prompt = input("\nEnter prompt: ") or "Tell me about the picture?"
        history.add_message(HumanMessage(prompt))
        response = llm.invoke(history.messages)
        history.add_message(AIMessage(response.content))
        print(textwrap.fill(response.content, width=100))


warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

if __name__ == "__main__":
    ChatSession().start()
    # multi_modal()
