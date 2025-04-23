# -----------------------------
# Import necessary libraries
# -----------------------------
import streamlit as st
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import requests  
from langchain_groq import ChatGroq

load_dotenv()

# -------------------------------------
# Initialize Arxiv and Wikipedia Tools
# -------------------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper,
                name="Wikipedia Search",
                description="Useful for finding general information about medical terms or drugs.")

pubmed = PubmedQueryRun(
    name="PubMed Search",
    description="Best for academic and clinical research articles related to medicines."
)

search = DuckDuckGoSearchRun(
    name="DuckDuckGo Web Search",
    description="Useful for general web results and current information about drugs or diseases."
)

class RxNormTool(BaseTool):
    name: str = "RxNorm Drug ID Lookup"
    description: str = "Use this to find standardized drug identifiers and information via RxNorm API. Input should be a drug name."

    def _run(self, query: str) -> str:
        try:
            url = f'https://rxnav.nlm.nih.gov/REST/rxcui.json?name={query}'
            response = requests.get(url)
            data = response.json()
            rxcui = data.get('idGroup', {}).get('rxnormId', [None])[0]
            if rxcui:
                details_url = f'https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json'
                details_response = requests.get(details_url)
                details_data = details_response.json()
                props = details_data.get('properties', {})
                name = props.get('name', 'Unknown')
                synonym = props.get('synonym', 'No synonyms available')
                return f"RxNorm ID: {rxcui}, Name: {name}, Synonyms: {synonym}"
            else:
                return "No RxNorm ID found."
        except Exception as e:
            return f"API Error: {str(e)}"

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented for RxNormTool.")

rxnorm_tool = RxNormTool()

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.title("🦜️🔗Agent for Medicine Suggestion and Medicine Function")
st.sidebar.title("Groq API KEY")
api_key = st.sidebar.text_input("Please Enter your Groq API Key:", type="password")

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to use the app.")
    st.stop()

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="Llama3-8b-8192",
    streaming=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Agent who can use medical tools to answer your query. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="Panadol"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    tools = [pubmed, rxnorm_tool, arxiv, search, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )
#   Rephrase this medical query for better search accuracy. Make it concise, factual, User query: "{original_prompt}"
#   and easy for a search tool to handle.

    def clarify_prompt(original_prompt, llm):
        clarification_prompt = f"""
        You are a helpful assistant that improves user prompts for better medical searches.
        Here is a user query: "{original_prompt}"

        Classify the intent into one of these categories:
        1. Medicine suggestion (e.g., "What should I take for a cold?")
        2. Drug function/info (e.g., "What does Paracetamol do?")
        3. Academic or clinical articles (e.g., "Latest studies on cancer treatment")
        4. General info or news (e.g., "News about vaccine updates")

        Then rewrite the question clearly so a search tool can better respond.

        Respond in this format:
        Intent: <intent>
        Rephrased: <new prompt>
        """

        result = llm.invoke(clarification_prompt)
        lines = result.content.strip().split('\n')

        intent = "Unknown"
        new_prompt = original_prompt
        for line in lines:
            if line.lower().startswith("intent:"):
                intent = line.split(":", 1)[1].strip()
            elif line.lower().startswith("rephrased:"):
                new_prompt = line.split(":", 1)[1].strip()

        return intent, new_prompt

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            intent, new_prompt = clarify_prompt(prompt, llm)
            response = search_agent.run(new_prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}"

        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })

        st.write(prompt)
        st.write(response)
