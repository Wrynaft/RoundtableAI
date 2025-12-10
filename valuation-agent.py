import getpass
import os
from dotenv import load_dotenv

# Load environment variables
if load_dotenv():
    print("✅ Environment variables loaded successfully.")
else:
    print("⚠️ Could not load .env file. Ensure it's in the same directory.")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# For testing purposes
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You talk like a pirate. Answer all questions to the best of your ability."),
        MessagePlaceholder(variable_name="user_message")
    ]
)

from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessageState, StateGraph

workflow = StateGraph(state_schema=MessageState)

def call_model(state: MessageState):
    trimmed_messages = trimmer.invoke(state["messages"])
    print(f"Messages after trimming: {len(trimmed_messages)}")
    print("Remaining messages:")
    for msg in trimmed_messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"message": response}

workflow.add(START, "model")
workflow.add("model", call_model)

memory_saver = MemorySaver()

config = {"configurable": {"thread_id": "abc123"}}
app = workflow.compile(checkpointer=memory_saver)

config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

from langchain_core.messages import HumanMessage, AIMessage 

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="|")