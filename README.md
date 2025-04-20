# Medicine_Agent
Create a Langchain agent that suggest medicine on behalf of Symptoms and also define purpose of medicines 
# 🧠💊 Medicine Suggestion Agent with LangChain + Streamlit

A conversational agent powered by [LangChain](https://www.langchain.com/) and [Streamlit](https://streamlit.io/) that provides **medicine suggestions**, **drug functions**, and **medical literature** lookup using a combination of:

- 🔬 **PubMed** for biomedical research
- 📚 **ArXiv** and **Wikipedia** for general scientific context
- 🌐 **DuckDuckGo** for web search
- 💊 **RxNorm** API for retrieving drug RxCUI identifiers
- 🧠 **Groq's Llama 3** large language model for conversational intelligence

---

## 🚀 Features

- 💬 Conversational chat UI via Streamlit
- 🔍 Real-time research with PubMed, ArXiv, Wikipedia & DuckDuckGo
- 🧾 Drug lookup using RxNorm API
- 🦙 Powered by Groq Llama3 models
- 🔗 Tool-augmented agent using LangChain's `ZERO_SHOT_REACT_DESCRIPTION`

---

## 📸 Demo

![screenshot](assets/demo.png) <!-- Add screenshot path if available -->

---

## 🧩 Tech Stack

| Component       | Technology                      |
|----------------|----------------------------------|
| LLM             | Groq API (LLaMA 3)               |
| Framework       | [LangChain](https://www.langchain.com/) |
| Web UI          | [Streamlit](https://streamlit.io/) |
| External Tools  | RxNorm API, PubMed, Arxiv, Wikipedia, DuckDuckGo |

---

## 🔐 Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/medic-agent.git
cd medic-agent
Create and activate a virtual environment (optional)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables

Create a .env file in the root directory and add:

env
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
Alternatively, you can input the API key in the Streamlit sidebar manually.

🏃 Run the App
bash
Copy
Edit
streamlit run app.py
Then go to http://localhost:8501 in your browser.

📦 Tools Used

Tool	Purpose
PubmedQueryRun	Searches biomedical articles
RxNormTool	Looks up RxNorm IDs for drugs
ArxivQueryRun	Gets scientific papers from ArXiv
WikipediaQueryRun	Retrieves Wikipedia summaries
DuckDuckGoSearchRun	General-purpose web search
✨ Example Queries
"What is the function of ibuprofen?"

"Suggest medicine for seasonal allergies."

"What does paracetamol do?"

"Give research articles on diabetes management."

🧠 Powered By
LangChain

Groq

Streamlit

RxNorm API

PubMed

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgments
Thanks to the open-source community and medical data providers like PubMed and RxNorm for enabling this project.

