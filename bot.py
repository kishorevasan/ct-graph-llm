import streamlit as st
from utils import write_message
from agent import generate_response

# Page Config
st.set_page_config('TrialQuery:Clinical Trials Explorer',page_icon="💊", layout = 'wide',initial_sidebar_state="auto")

## style css
st.markdown(
    """
    <style>
    #content {
      display: none;
      font-size: 18px;
      margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>💊 TrialQuery: Clinical Trials Explorer</h1>
    <h2 style='text-align: center;'>Your AI-powered guide to tracking innovation in clinical trials.</h2>

    <p style="font-size: 18px; max-width: 800px; margin: auto; text-align: center;">
    Stay on top of the innovation pipeline with real-time data. TrialQuery lets you ask natural questions like:
    <i>"What are the ongoing trials for diabetes?"</i> or <i>"How many Diabetes trials are funded by GlaxoSmithKline, and what is the summary of those trials?"</i>

    Answering these kinds of questions usually requires complex SQL and access to massive real-time datasets. TrialQuery simplifies this by combining graph-based reasoning with semantic search in a Hybrid RAG (Retrieval-Augmented Generation) approach. It leverages a curated clinical trials knowledge graph to minimize hallucinations and improve accuracy.
    \n Give it a try - and explore how AI can make clinical research more transparent and accessible. As always, the app is <strong>open source</strong> and reproducible: <a href="https://github.com/kishorevasan/ct-graph-llm" target="_blank">View on GitHub</a>
    <hr style='margin-top: 0px;'>
    """, unsafe_allow_html=True
)

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the Clinical Trials Explorer! How can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        ## call LLM
        response = generate_response(message)
        write_message('assistant', response)

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Chat input
#st.markdown("---")
prompt = st.chat_input("💬 Ask me about clinical trials...", key="custom-input")

# Handle any user input
if prompt:
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
