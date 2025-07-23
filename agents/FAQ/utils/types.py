from typing import Literal, TypedDict
from langgraph.graph import MessagesState

class Message(TypedDict, total=False):
  role: Literal['user', 'human', 'system', 'assistant']
  content: str
  name: str

class IterativeMessageState(MessagesState):
  retry: str
