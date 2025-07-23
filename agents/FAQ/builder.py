from agents.FAQ.faq_customer_service import (
  response_composer,
  retrieve_from_vector,
  query,
  paraphrase_and_retry,
  should_retry
)
from .utils.types import IterativeMessageState

# from utils.types import State
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

def build_graph():
  builder = StateGraph(IterativeMessageState)

  # define all nodes
  # builder.add_node('RetrieveFromVector', retrieve_from_vector)
  builder.add_node('Query', query)
  builder.add_node('Tool', ToolNode([retrieve_from_vector]))
  builder.add_node('ResponseComposer', response_composer)
  builder.add_node('ParaphraseAndRetry', paraphrase_and_retry)

  # connect all nodes with edges
  builder.add_edge(START, 'Query')
  builder.add_conditional_edges(
    'Query',
    tools_condition,
    { END: END, 'tools': 'Tool' },
  )
  builder.add_edge('Tool', 'ResponseComposer')
  builder.add_conditional_edges(
    'ResponseComposer',
    should_retry,
    { 'end': END, 'paraphrase': 'ParaphraseAndRetry' }
  )
  builder.add_edge('ParaphraseAndRetry', 'Query')

  return builder