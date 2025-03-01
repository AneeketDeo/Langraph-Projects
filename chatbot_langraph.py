from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
import matplotlib.pyplot as plt




# CREATE STATE GRAPH
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


# ADD NODES

llm = ChatOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",

)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# ADD START 
# Add edge  
graph_builder.add_edge(START, "chatbot")

# ADD EXIT / END
graph_builder.add_edge("chatbot", END)

# RUN THE GRAPH
graph = graph_builder.compile()

# DRAW THE GRAPH

try:
    graph_img = Image(graph.get_graph().draw_mermaid_png()) 
    display(graph_img)
    with open("graph_output.png", "wb") as f:
        f.write(graph_img)
except Exception:
    # This requires some extra dependencies and is optional
    pass


# MAIN LOOP
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break