from langchain.chat_models import init_chat_model

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from .models.builder import load_vector_db
from .utils.types import IterativeMessageState

from dotenv import load_dotenv
load_dotenv(override=True)

llm = init_chat_model('gpt-4.1-mini', model_provider='openai')
MAX_ITERATION = 2

@tool
def retrieve_from_vector(query: str):
  """Retrieve relevant keywords and search results from vector database based on user's questions."""
  top_k = 7 # TODO: widen search space if on retry
  vector_db = load_vector_db('data', 'mini_project_vector')

  # Perform a similarity search in the vector database
  results = vector_db.similarity_search(query, k=top_k)
  joined = '\n\n'.join([doc.page_content for doc in results])

  return joined

def should_retry(state: IterativeMessageState) -> bool:
  retry = state.get('retry', 0)

  if 0 < retry < MAX_ITERATION:
    return 'paraphrase'

  return 'end'

# Nodes
def query(state: IterativeMessageState) -> IterativeMessageState:
  """Query the vector database and return the results."""
  llm_tool = llm.bind_tools([retrieve_from_vector])

  prompt = '''
    Carilah informasi HANYA menggunakan tools yang sudah diberikan.
    Gunakan HANYA informasi dari tools berikut untuk menjawab.
    Jika tidak ada hasil dari tools, jangan berimprovisasi.
  '''
  response = llm_tool.invoke([SystemMessage(content=prompt)] + state['messages'])

  return { 'messages': [response] }

def response_composer(state: IterativeMessageState) -> IterativeMessageState:
  """Arrange the tool messages in the response into an input for prompt."""
  if 'retry' not in state:
    state['retry'] = 0

  # Extract the response from the state
  messages = state['messages']

  human_message = messages[-3].content if len(messages) > 2 else ''
  tool_message = messages[-1].content if messages else ''

  prompt = f'''
    Anda adalah seorang CS yang dapat menjawab pertanyaan berdasarkan informasi yang relevan dari hasil pencarian.
    Jawablah pertanyaan dengan singkat. Jika tidak ada informasi yang relevan,
    katakan "Mohon maaf, saya tidak tahu. Silakan ajukan pertanyaan lainnya."

    Berikut informasi untuk menjawab pertanyaan:
    {tool_message}
  '''

  # print('Prompt:', prompt)
  # print('Human Message:', human_message)

  response = llm.invoke([
    SystemMessage(content=prompt),
    HumanMessage(content=human_message)
  ])

  if response.content == 'Mohon maaf, saya tidak tahu. Silakan ajukan pertanyaan lainnya.':
    state['retry'] += 1

  if 0 < state['retry'] < MAX_ITERATION:
    return { 'messages': messages, 'retry': state['retry'] }

  return { 'messages': [response], 'retry': state['retry'] }

def paraphrase_and_retry(state: IterativeMessageState) -> IterativeMessageState:
  """Paraphrase the question and retry the query."""
  messages = state['messages']
  prompt = f'''
    Anda bertugas memparafrase sebuah kalimat tanya atau perintah.
    Gunakan diksi yang berbeda dari kalimat tersebut namun tetap pertahankan maknanya.
  '''

  response = llm.invoke([
    SystemMessage(content=prompt),
    HumanMessage(content=messages[0].content)
  ])

  # print('Paraphrased question:', response.content)

  return { 'messages': [HumanMessage(content=response.content)], 'retry': state['retry'] }
