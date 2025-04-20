# -----------------------------
# Import necessary libraries
# -----------------------------
import streamlit as st  # Streamlit for building the web app
from langchain_groq import ChatGroq  # Groq-based Chat LLM
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  # Utilities to query Arxiv and Wikipedia
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun  # Tools for each wrapper
from langchain.agents import initialize_agent, AgentType  # Langchain for agent initialization
from langchain.callbacks import StreamlitCallbackHandler  # Callback handler to stream responses in Streamlit
import os
from dotenv import load_dotenv  # For environment variables (if needed)
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import requests  



# -----------------------------
# Initialize Arxiv and Wikipedia Tools
# -----------------------------

# Create an Arxiv wrapper with settings for how many results to return
# and how many characters to retrieve from each article
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Create a Wikipedia wrapper with settings for how many results to return
# and how many characters to retrieve from each article
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Create a DuckDuckGo tool for general web search
search = DuckDuckGoSearchRun(name="Search")

# Create a PubMed tool for searching medical literature
pubmed = PubmedQueryRun()

# Creat e a custom tool for RxNorm API  
class RxNormTool(BaseTool):
    name: str = "RxNorm_Search"
    description: str = "Finds RxNorm IDs for drugs. Input: drug name."

    def _run(self, query: str) -> str:
        try:
            url = f'https://rxnav.nlm.nih.gov/REST/rxcui.json?name={query}'
            response = requests.get(url)
            data = response.json()
            rxcui = data.get('idGroup', {}).get('rxnormId', [None])[0]
            return rxcui or "No RxNorm ID found."
        except Exception as e:
            return f"API Error: {str(e)}"

rxnorm_tool = RxNormTool()

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.title("🦜️🔗Agent for Medicine Suggestion and Medicine Function")  # Title of the Streamlit app

# -----------------------------
# Sidebar for settings
# -----------------------------
st.sidebar.title("Groq API KEY")  # Sidebar title
api_key = st.sidebar.text_input("Please Enter your Groq API Key:", type="password")  # Input for Groq API key

# -----------------------------
# Session State for Chat Messages
# -----------------------------
# If there's no "messages" key in st.session_state, initialize it
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?"
        }
    ]

# Display all existing messages in the chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# -----------------------------
# Chat Input and Processing
# -----------------------------
# Read user input from the chat input box
if prompt := st.chat_input(placeholder="What is Machine learning?"):
    # Store user's message in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user's message in chat
    st.chat_message("user").write(prompt)

    # Initialize the ChatGroq LLM with the provided Groq API key
    # and specify the LLM model name, along with streaming = True
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True
    )
    
    # A list of tools that the agent can use
    tools = [pubmed, rxnorm_tool, search, arxiv, wiki]
    # Create a Langchain agent with the ZERO_SHOT_REACT_DESCRIPTION strategy
    # The agent will attempt to parse the user's query and decide whether
    # to use the tools
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors = True,
        verbose=True,
    )

    # -----------------------------
    # Get and Display the Response
    # -----------------------------
    with st.chat_message("assistant"):
        # Create a Streamlit callback handler so we can display the thought process
        # The 'expand_new_thoughts' parameter controls how the chain of thought is expanded
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Run the agent with the collected messages and use the callback
        try:
            response = search_agent.run(prompt, callbacks=[st_cb])  # Pass only the latest query
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}"
        
        # Store the assistant's response in session state
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })
        
        # Finally, display the assistant's response
        st.write(response)
