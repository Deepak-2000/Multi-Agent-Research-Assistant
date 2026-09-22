import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Initialize Groq LLM
# ============================================================

llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0)

# ============================================================
# PDF QA Chain
# ============================================================

qa_chain = None

uploaded_file = st.file_uploader("Upload PDF",type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector database
    vectordb = Chroma.from_documents(chunks,embeddings)

    # Retriever
    retriever = vectordb.as_retriever()

    # Prompt for PDF QA
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant.

                Answer the user's question using the
                provided context.

                If the answer cannot be found in the
                context, clearly say that the information
                is not available in the uploaded document.

                Context:
                {context}
                """
            ),
            (
                "human",
                "{input}"
            )
        ]
    )

    # Create document chain
    stuff_chain = create_stuff_documents_chain(llm,prompt)

    # Create retrieval chain 
    qa_chain = create_retrieval_chain(retriever,stuff_chain)
    st.success(f"PDF uploaded successfully: {uploaded_file.name}")


# ============================================================
# PDF Tool
# ============================================================

@tool
def query_pdf(query: str) -> str:
    """
    Search the uploaded PDF and answer questions
    based on its contents.
    """

    if qa_chain is None:
        return "No PDF has been uploaded."

    try:

        result = qa_chain.invoke(
            {
                "input": query
            }
        )

        return result["answer"]

    except Exception as e:

        return f"Error while querying PDF: {str(e)}"


# ============================================================
# Web Search Tool
# ============================================================

search_tool = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """
    Search the web for current, recent, or
    up-to-date information.
    """

    try:

        result = search_tool.run(query)

        return result

    except Exception as e:

        return f"Error while searching the web: {str(e)}"


# ============================================================
# Tools
# ============================================================

tools = [
    query_pdf,
    web_search
]


# ============================================================
# Create Agent
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
    You are a research assistant.

    You have access to two tools:

    1. query_pdf
       Use this tool when the user asks about information
       contained in the uploaded PDF.

    2. web_search
       Use this tool when the user asks for:
       - latest information
       - current information
       - recent information
       - information from the internet
       - news
       - information that may have changed recently

    For general knowledge and reasoning questions,
    answer directly without using a tool.

    Always choose the appropriate tool when necessary.

    After receiving the tool result, provide a clear,
    concise and useful final answer.
    """
)


# ============================================================
# Streamlit UI
# ============================================================

st.title(
    "🤖 Multi-Agent Research Assistant"
)

st.write("""
    Ask questions about your uploaded PDF or ask
    questions that require web research.
    """
)

query = st.text_input("Ask a question:")


# ============================================================
# Execute Agent
# ============================================================

if query:

    with st.spinner("Researching..."):

        try:

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                }
            )


            # Get final message
            final_message = result["messages"][-1]


            st.subheader("Answer")

            st.write(
                final_message.content
            )


        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )