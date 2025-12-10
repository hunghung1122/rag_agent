from typing import Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

class FunctionCall(TypedDict):
    name: Literal["retrieve"]
    arguments: str

class AgentAction(TypedDict):
    type: Literal["function_call", "finish"]
    function_call: Optional[FunctionCall]

class State(TypedDict, total=False):
    query: str
    rewritten_query: str
    documents: list
    answer: str
    agent_action: AgentAction
    relevance: Literal["yes","no"]

model = ChatOllama(model="gpt-oss:120b-cloud")
embed = OllamaEmbeddings(model="moondream:latest")
vec = Chroma(persist_directory="vector_store", embedding_function=embed)
retriever = vec.as_retriever(search_kwargs={"k":4})

def agent(state: State):
    if "documents" not in state:
        return {"agent_action": {"type":"function_call","function_call":{"name":"retrieve","arguments":state["query"]}}}
    return {"agent_action": {"type":"finish","function_call":None}}

def should_retrieve(state: State):
    return state["agent_action"]["type"] == "function_call"

def tool(state: State):
    q = state.get("rewritten_query", state["query"])
    docs = retriever.invoke(q)
    return {"documents": docs}

def check_relevance(state: State):
    q = state.get("rewritten_query", state["query"])
    txt = " ".join(d.page_content for d in state["documents"])
    r = model.invoke(f"Query: {q}\nDocs: {txt}\nRelevant? yes or no.").content.lower()
    return {"relevance": "yes" if "yes" in r else "no"}

def relevance_route(state: State):
    return state["relevance"]

def rewrite(state: State):
    r = model.invoke(f"Rewrite to improve retrieval: {state['query']}").content
    return {"rewritten_query": r}

def generate(state: State):
    txt = "\n".join(d.page_content for d in state["documents"])
    a = model.invoke(f"Docs:\n{txt}\nAnswer: {state['query']}").content
    return {"answer": a}

graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tool", tool)
graph.add_node("check_relevance", check_relevance)
graph.add_node("rewrite", rewrite)
graph.add_node("generate", generate)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_retrieve, {True:"tool", False:END})
graph.add_edge("tool", "check_relevance")
graph.add_conditional_edges("check_relevance", relevance_route, {"yes":"generate","no":"rewrite"})
graph.add_edge("rewrite", "tool")
graph.add_edge("generate", END)

app = graph.compile()

if __name__ == "__main__":
    print(" Testing Graph ")
    for event in app.stream({"query": "Who is mian usman?"}):
        print("STEP:", event)
