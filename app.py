import streamlit as st
import asyncio
from work_flow import create_workflow

st.title("InsureGenius : Your AI Assistant for Insurance Claims")

workflow = create_workflow()

async def invoke_wf(message):
    return await workflow.run(message=message)

with st.sidebar:
    st.image("images/image.png", width=500)
    add_radio = st.radio(
        "What can I do for you today?",
        ("/chat with the knowledge base",
         "/show me the workflow diagram!",)
    )

if add_radio == "/chat with the knowledge base":
    st.header("Chat with the knowledge base")
    st.write("Please type your message below:")
    st.write("Example_1 : Whats the fpp critical illness underwriting limits for 51-55 olds?")
    st.write("Example-2: Show me member details for member with name John Doe?")
    st.write("Example-3: Show me the last claim details for member with name Charlie Davis?")
    request = st.text_area("", height=100)
    submit = st.button("submit", type="primary")

    if submit and request:
        result = asyncio.run(invoke_wf(request))
        st.write(result)

elif add_radio == "/show me the workflow diagram!":
    st.image("images/workflow.png", width=800)