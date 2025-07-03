__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from markitdown import MarkItDown
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import create_openai_tools_agent
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
import uuid
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile

groq_api_key = st.secrets["groq_api_key"]
os.environ['HF_TOKEN'] = st.secrets["hf_token"]
os.environ['LANGCHAIN_API_KEY'] = st.secrets["langchain_api_key"]
os.environ['LANGCHAIN_PROJECT'] = st.secrets["langchain_project"]
os.environ['LANGCHAIN_TRACING_V2'] = "true"

st.title("📚 Document Ingestion & Retrieval App | AI Agent")
st.info("👋 Welcome! In the sidebar, please select either the RAG mode for document ingestion and retrieval or the AI agent mode for wikipedia, arxiv, and/or web search.")
    
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id
st.sidebar.markdown(f"#️⃣ **Session ID:** `{session_id}`")

def init_state(key, deault):
    if key not in st.session_state:
        st.session_state[key] = default

for key, default in [
    ("agent_mode", False),
    ("rag_mode", False),
    ("docs_uploaded", False),
    ("tools", []),
]:
    init_state(key, default)

mode = st.sidebar.radio("Select Mode", ["📄 RAG", "🤖 AI Agent"])
st.session_state.rag_mode = mode == "📄 RAG"
st.session_state.agent_mode = mode == "🤖 AI Agent"

col1, col2 = st.columns(2)
with col1:
    if st.session_state.rag_mode:
        st.session_state.rag_mode = True
        st.session_state.agent_mode = False

with col2:
    if st.session_state.agent_mode:
        st.session_state.rag_mode = False
        st.session_state.agent_mode = True

tab1, tab2 = st.tabs(["📄 RAG Mode", "🤖 AI Agent Mode"])
with tab1:
    if st.session_state.rag_mode:
        # Data Ingestion
        md = MarkItDown(enable_plugins=False)
        st.subheader("Step 1: Upload Your Documents")
        st.markdown("Upload up to **3 PDFs**. Press *Process Documents* when ready.")
        uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True, key="file_uploader")
        if uploaded_files:
            loaders = []
            for i in range(len(uploaded_files)):
                if i <= 2:
                    temppdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                    with open(temppdf, "wb") as file:
                        file.write(uploaded_files[i].getvalue())

                    md_file = md.convert(temppdf)
                    tempmd = tempfile.NamedTemporaryFile(delete=False, suffix=".md").name
                    with open(tempmd, "w") as file:
                        file.write(md_file.markdown)
                    loader = UnstructuredMarkdownLoader(tempmd)
                    loaders.append(loader)

            if uploaded_files and st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    loader_all = MergedDataLoader(loaders=loaders)
                    docs = loader_all.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    docs_split = splitter.split_documents(docs)
                    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
                    vectorstore = Chroma.from_documents(documents=docs_split, embedding=embedding_model, persist_directory="./db")
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) 
                    st.session_state.retriever = retriever
                    st.success("Documents have been successfully ingested.")
                    st.session_state.docs_uploaded = True

        if st.session_state.docs_uploaded:

            llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

            contextualize_q_system_prompt=(
                '''Given a chat history and the latest user question
                which might reference context in the chat history, 
                formulate a standalone question which can be understood 
                without the chat history. Do NOT answer the question, 
                just reformulate it if needed and otherwise return it as is.
            '''
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)

            system_prompt = (
                    "You are a helpful assistant. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Keep the answer concise. "
                    "Always return the numbers if the question asks about values."
                    "\n\n"
                    "{context}"
                )
            qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if "chat_histories" not in st.session_state:
                    st.session_state.chat_histories = {}
                if session not in st.session_state.chat_histories:
                    st.session_state.chat_histories[session] = ChatMessageHistory()
                return st.session_state.chat_histories[session]

            conversational_rag_chain=RunnableWithMessageHistory(
                rag_chain, 
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            st.subheader("Step 2: Ask Your Questions")
            st.text_input("Ask a question to the RAG application:", key="user_input")
            if st.button("Submit"):
                with st.spinner("Retrieving information..."):
                    user_input = st.session_state.user_input
                    if user_input:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={
                                "configurable": {"session_id":session_id}
                            },
                        )
                        st.write(response['answer'])

            chat_history = get_session_history(session_id).messages
            st.markdown("### Chat History")
            with st.expander("Chat History"):
                for msg in chat_history:
                        with st.chat_message("user" if msg.type == "human" else "assistant"):
                            st.markdown(msg.content)

            if st.button("Clear Chat History"):
                with st.spinner("Clearing chat history..."):
                    if "chat_histories" in st.session_state:
                        st.session_state.chat_histories.pop(session_id, None)
                    st.rerun()


with tab2:
    if st.session_state.agent_mode:
        # Tools
        st.subheader("Step 1: Select Your Tools")
        if st.button("Wikipedia Search"):
            api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
            wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
            if wiki not in st.session_state.tools:
                st.session_state.tools.append(wiki)

        if st.button("Arxiv Search"):
            api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
            arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
            if arxiv not in st.session_state.tools:
                st.session_state.tools.append(arxiv)

        if st.button("Web Search"):
            search = DuckDuckGoSearchRun(name="web search")
            if search not in st.session_state.tools:
                st.session_state.tools.append(search)

        st.sidebar.write("Selected Tools:")
        for tool in st.session_state.tools:
            st.sidebar.write("-", tool.name)

        # LLM
        llm = ChatGroq(model="llama3-70b-8192", temperature=0.3, groq_api_key=groq_api_key)

        # Agent
        if st.session_state.tools:
            prompt = hub.pull("hwchase17/openai-functions-agent")
            agent = create_openai_tools_agent(llm, st.session_state.tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=st.session_state.tools, verbose=True)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if "chat_histories" not in st.session_state:
                    st.session_state.chat_histories = {}
                if session not in st.session_state.chat_histories:
                    st.session_state.chat_histories[session] = ChatMessageHistory()
                return st.session_state.chat_histories[session]

            conversational_agent_chain = RunnableWithMessageHistory(
                agent_executor, 
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="output"
            )

            st.subheader("Step 2: Ask Your Questions")
            st.text_input("Ask a question to the AI agent:", key="user_input")
            if st.button("Submit"):
                with st.spinner("AI agent processing information..."):
                    user_input = st.session_state.user_input
                    if user_input:
                        response = conversational_agent_chain.invoke(
                            {"input": user_input},
                            config={
                                "configurable": {"session_id": session_id}
                            },
                        )
                        st.write(response['output'])

            chat_history = get_session_history(session_id).messages
            st.markdown("### Chat History")
            with st.expander("Chat History"):
                for msg in chat_history:
                        with st.chat_message("user" if msg.type == "human" else "assistant"):
                            st.markdown(msg.content)

            if st.button("Clear Chat History"):
                with st.spinner("Clearing chat history..."):
                    if "chat_histories" in st.session_state:
                        st.session_state.chat_histories.pop(session_id, None)
                    st.rerun()