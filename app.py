import streamlit as st
import tempfile
import hashlib

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

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_classic.chains import create_retrieval_chain


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="Multi-PDF Research Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Session State
# ============================================================

if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

if "pdf_hash" not in st.session_state:
    st.session_state["pdf_hash"] = None

if "pdf_names" not in st.session_state:
    st.session_state["pdf_names"] = []

if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None


# ============================================================
# Groq LLM
# ============================================================

try:

    # llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=st.secrets["GROQ_API_KEY"])
    llm = ChatGroq(model="openai/gpt-oss-20b",groq_api_key=st.secrets["GROQ_API_KEY"])

except Exception as e:

    st.error("Unable to initialize Groq LLM.\n\n" "Please check your GROQ_API_KEY in Streamlit Secrets.")
    st.stop()


# ============================================================
# Embeddings
# ============================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("📚 PDF Knowledge Base")


uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"],accept_multiple_files=True)


# ============================================================
# Create Unique Hash for Uploaded PDFs
# ============================================================

def calculate_files_hash(files):

    hasher = hashlib.md5()

    for file in files:

        hasher.update(file.name.encode("utf-8"))

        hasher.update(file.getvalue())

    return hasher.hexdigest()


# ============================================================
# Process Multiple PDFs
# ============================================================

if uploaded_files:

    current_hash = calculate_files_hash(uploaded_files)

    # Process only when files change
    if current_hash != st.session_state["pdf_hash"]:
        with st.spinner("Processing uploaded PDF files..."):
            try:

                all_documents = []
                pdf_names = []

                # ------------------------------------------------
                # Load every PDF
                # ------------------------------------------------

                for uploaded_file in uploaded_files:

                    pdf_names.append(uploaded_file.name)

                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name

                    # Load PDF
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    # Add filename metadata
                    for document in documents:
                        document.metadata["source"] = uploaded_file.name

                    all_documents.extend(documents)

                # ------------------------------------------------
                # Split all PDFs
                # ------------------------------------------------

                splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
                chunks = splitter.split_documents(all_documents)

                # ------------------------------------------------
                # Create unique temporary vector DB
                # ------------------------------------------------

                persist_dir = tempfile.mkdtemp()

                # ------------------------------------------------
                # Create Chroma DB
                # ------------------------------------------------

                vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)

                # ------------------------------------------------
                # Store Vector DB in Session State
                # ------------------------------------------------

                st.session_state["vector_db"] = (vector_db)


                # ------------------------------------------------
                # Create Retriever
                # ------------------------------------------------

                retriever = vector_db.as_retriever(search_kwargs={"k": 5})


                # =================================================
                # PDF Prompt
                # =================================================

                pdf_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
                            You are a document question-answering
                            assistant.

                            You have access to multiple uploaded
                            PDF documents.

                            Answer the user's question using the
                            information contained in the provided
                            context.

                            Important rules:

                            1. Do not invent information.

                            2. If the answer is not available in
                               the documents, clearly say that the
                               information is not available in the
                               uploaded PDFs.

                            3. When possible, mention the source
                               PDF filename from the metadata.

                            4. If information comes from multiple
                               PDFs, combine the relevant
                               information clearly.

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


                # ------------------------------------------------
                # Document Chain
                # ------------------------------------------------

                stuff_chain = create_stuff_documents_chain(llm,pdf_prompt)

                # ------------------------------------------------
                # Retrieval Chain
                # ------------------------------------------------

                qa_chain = create_retrieval_chain(retriever,stuff_chain)

                # ------------------------------------------------
                # Store QA Chain
                # ------------------------------------------------

                st.session_state["qa_chain"] = (qa_chain)

                st.session_state["pdf_hash"] = (current_hash)

                st.session_state["pdf_names"] = (pdf_names)


                # ------------------------------------------------
                # Success
                # ------------------------------------------------

                st.sidebar.success("PDFs processed successfully!")
                st.sidebar.write(f"📚 Documents: {len(pdf_names)}")

                st.sidebar.write(f"📄 Pages: {len(all_documents)}")

                st.sidebar.write(f"🧩 Chunks: {len(chunks)}")

            except Exception as e:

                st.error(f"Error while processing PDFs:\n\n{str(e)}")

                st.session_state["qa_chain"] = None

                st.session_state["vector_db"] = None

                st.session_state["pdf_hash"] = None


# ============================================================
# Display Uploaded Files
# ============================================================

if st.session_state["pdf_names"]:

    st.sidebar.divider()

    st.sidebar.subheader("📚 Uploaded PDFs")

    for name in st.session_state["pdf_names"]:
        st.sidebar.write(f"📄 {name}")


# ============================================================
# PDF Tool
# ============================================================

@tool
def query_pdf(query: str) -> str:
    """
    Search all uploaded PDFs and answer the user's
    question using their contents.
    """

    qa_chain = st.session_state.get("qa_chain")

    if qa_chain is None:

        return ("No PDF documents have been uploaded. "
            "Please upload one or more PDFs first."
        )

    try:
        result = qa_chain.invoke({"input": query})
        answer = result.get("answer",None)
        if not answer:
            return (
                "No relevant answer was found "
                "in the uploaded PDFs."
            )

        return answer


    except Exception as e:
        return (
            f"Error while querying the PDFs: {str(e)}"
        )

# if st.button("Debug Retriever"):

#     vector_db = st.session_state.get("vector_db")

#     if vector_db is None:

#         st.error("Vector DB is None")

#     else:

#         docs = vector_db.similarity_search(
#             "workflow in Aras",
#             k=10
#         )

#         st.subheader("Retrieved Documents")

#         st.write(f"Retrieved {len(docs)} chunks")

#         for i, doc in enumerate(docs):

#             st.markdown(f"### Chunk {i + 1}")

#             st.write(
#                 "Source:",
#                 doc.metadata.get("source", "Unknown")
#             )

#             st.write(
#                 "Page:",
#                 doc.metadata.get("page", "Unknown")
#             )

#             st.write(doc.page_content[:2000])

#             st.divider()
# ============================================================
# Web Search Tool
# ============================================================

search_tool = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current, recent,
    or up-to-date information.
    """

    try:
        result = search_tool.run(query)
        return result

    except Exception as e:
        return (
            f"Error while searching the web: {str(e)}"
        )


# ============================================================
# Tools
# ============================================================

tools = [query_pdf,web_search]


# ============================================================
# Research Agent
# ============================================================

agent = create_agent(model=llm,tools=tools,
    system_prompt="""
    You are a research assistant.

    You have access to two tools.

    ============================================================
    1. query_pdf
    ============================================================

    Use query_pdf when the user's question is about
    the uploaded PDF documents.

    There may be multiple PDFs.

    The tool searches across all uploaded PDFs.

    Examples:

    - "What does the documentation say about workflows?"
    - "Explain the lifecycle described in the documents."
    - "What is mentioned about Aras Forms?"
    - "Compare the information from the uploaded documents."

    ============================================================
    2. web_search
    ============================================================

    Use web_search when the user asks for:

    - Latest information
    - Current information
    - Recent information
    - News
    - Internet information
    - Information that may have changed recently

    ============================================================
    GENERAL QUESTIONS
    ============================================================

    For general knowledge and reasoning questions,
    answer directly without using a tool.

    ============================================================
    TOOL SELECTION
    ============================================================

    If the question refers to uploaded PDFs,
    use query_pdf.

    If the question requires current internet information,
    use web_search.

    If the question is general knowledge,
    answer directly.

    After receiving the tool result, provide a clear,
    concise and useful answer.

    Do not expose internal tool execution details.
    """
)


# ============================================================
# Main UI
# ============================================================

st.title("🤖 Multi-PDF Research Assistant")

st.write("""
    Upload multiple PDF documents and ask questions
    across all of them. You can also ask questions
    that require web research.
    """)

# ============================================================
# PDF Status
# ============================================================

if st.session_state["qa_chain"] is not None:

    st.info(
        f"📚 {len(st.session_state['pdf_names'])} "
        f"PDF document(s) are ready for questions."
    )

else:

    st.warning(
        "📚 No PDFs uploaded. "
        "You can still ask general or web-based questions."
    )


# ============================================================
# User Query
# ============================================================

query = st.text_input("Ask a question:",placeholder=("Example: described in the uploaded PDFs."))


# ============================================================
# Execute Agent
# ============================================================

if query:

    with st.spinner("Researching..."):

        try:

            # ====================================================
            # If PDFs are uploaded, use PDF RAG directly
            # ====================================================

            qa_chain = st.session_state.get("qa_chain")

            if qa_chain is not None:

                result = qa_chain.invoke({
                    "input": query
                })

                answer = result.get("answer", "")

                st.subheader("Answer")

                st.write(answer)

                # ------------------------------------------------
                # Show retrieved documents
                # ------------------------------------------------

                with st.expander("📚 Retrieved Documents"):

                    documents = result.get("context", [])

                    st.write(
                        f"Retrieved {len(documents)} chunks"
                    )

                    for i, doc in enumerate(documents, 1):

                        st.markdown(
                            f"### Chunk {i}"
                        )

                        st.write(
                            f"**Source:** "
                            f"{doc.metadata.get('source', 'Unknown')}"
                        )

                        st.write(
                            f"**Page:** "
                            f"{doc.metadata.get('page', 'Unknown')}"
                        )

                        st.write(doc.page_content)

                        st.divider()

            # ====================================================
            # No PDFs → use Research Agent
            # ====================================================

            else:

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

                messages = result.get(
                    "messages",
                    []
                )

                if not messages:

                    st.error(
                        "The agent did not return a response."
                    )

                else:

                    final_message = messages[-1]

                    st.subheader("Answer")

                    st.write(
                        final_message.content
                    )

        except Exception as e:

            st.error(
                f"Error while processing the question:\n\n{str(e)}"
            )
