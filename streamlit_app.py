# app.py
"""
Streamlit Multi-Agent RAG using Google Gemini (preferred) or HuggingFace embeddings (fallback).

Changes in this version:
- No sidebar or file uploads.
- Salary.txt and Insurance.txt are embedded below.
- Vector stores are built automatically on app start.

"""

from __future__ import annotations
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import streamlit as st

# LangChain pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Vector store
from langchain_community.vectorstores import FAISS

# Embeddings: Gemini preferred, HF fallback
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    _HAVE_GEMINI = True
except Exception:
    _HAVE_GEMINI = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

# ---------- Config / Keywords ----------
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

SALARY_KEYWORDS = {
    "salary", "pay", "wage", "ctc", "compensation", "annual", "monthly", "bonus", "increment",
    "deduction", "allowance", "overtime", "net", "gross", "package", "in-hand", "annual salary",
}
INSURANCE_KEYWORDS = {
    "insurance", "policy", "premium", "claim", "coverage", "deductible", "benefit", "sum insured",
    "network", "cashless", "endorsement", "co-pay", "rider", "exclusions", "renewal",
}
# os.environ["GOOGLE_API_KEY"] = "API_KEY"

# ---------- Built-in documents (replaces uploads) ----------
BUILTIN_FILES: List[Tuple[str, str]] = [
    (
        "Salary.txt",
        """Salary refers to the regular payment an employee receives for their work. It is typically defined as a fixed monthly amount but can also be expressed annually. 

1. Monthly Salary:
   - This is the base pay an employee receives each month before any deductions.
   - For example, if your monthly salary is $4,000, this is your gross monthly income.

2. Annual Salary:
   - The annual salary is calculated by multiplying the monthly salary by 12.
   - Using the previous example, $4,000 × 12 = $48,000 per year.
   - Some organizations may also include bonuses and allowances in the annual salary.

3. Deductions:
   - Deductions are amounts subtracted from gross salary before arriving at net salary.
   - Common deductions include:
     - Income tax
     - Social security contributions
     - Retirement or pension contributions
     - Health insurance premiums (if shared by employer and employee)

4. Net Salary:
   - Net salary (also called take-home pay) is the amount received after all deductions.
   - Example: If gross monthly salary is $4,000 and deductions total $800, then net salary is $3,200.

5. Salary Structure:
   - Basic Pay: The core salary component.
   - Allowances: Housing allowance, transport allowance, meal allowance, etc.
   - Bonuses: Performance-based or annual bonuses.
   - Other Benefits: Stock options, overtime pay, or special incentives.

Overall, salary represents both fixed and variable components, and understanding deductions is important for accurate financial planning."""
    ),
    (
        "Insurance.txt",
        """Insurance benefits are a key part of an employee’s compensation package. They provide financial protection for medical and other unexpected expenses. 

1. Coverage:
   - Typical insurance coverage includes hospitalization, medical treatment, emergency care, and surgery.
   - Some plans also cover maternity care, dental care, and vision care.
   - Many employers extend coverage to dependents such as spouse and children.

2. Premiums:
   - The insurance premium is the cost of the insurance plan.
   - Employers often share this cost with employees.
   - Example: If the monthly premium is $500, the employer might pay $350 while the employee pays $150.

3. Claim Process:
   - To file a claim, employees must submit a claim form along with supporting documents such as medical bills and hospital reports.
   - The insurance provider reviews the claim and reimburses eligible expenses.
   - Some policies offer cashless hospitalization, where the insurer pays the hospital directly.

4. Exclusions:
   - Certain treatments or conditions may not be covered, such as cosmetic surgery or pre-existing conditions (within the first year of the policy).
   - It is important for employees to review policy documents to understand exclusions.

5. Additional Benefits:
   - Some insurance plans include wellness programs, annual health check-ups, mental health support, and telemedicine services.
   - Group insurance policies may offer higher coverage limits at lower premiums compared to individual plans.

In summary, insurance benefits reduce financial stress for employees by covering major medical costs and supporting their families in times of need."""
    ),
]

# ---------- Simple dataclass-like container ----------
class DomainStores:
    def __init__(self, salary: Optional[FAISS] = None, insurance: Optional[FAISS] = None):
        self.salary = salary
        self.insurance = insurance

# ---------- Embeddings + LLM helpers (explicit API key to avoid ADC error) ----------
def get_embeddings():
    """
    Return embeddings object.
    Prefer Google Gemini embeddings if GOOGLE_API_KEY is present; else HuggingFace fallback.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    # If you prefer Gemini embeddings, uncomment these two lines and ensure the package is installed.
    # if _HAVE_GEMINI and api_key:
    #     return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    if _HAVE_HF:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    raise RuntimeError(
        "No embeddings provider available. Install langchain-google-genai and set GOOGLE_API_KEY, "
        "or install sentence-transformers."
    )

def get_llm(temperature: float = 0.0):
    """
    Return Gemini chat LLM if available and API key present, else return None for fallback.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if _HAVE_GEMINI and api_key:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=temperature, google_api_key=api_key)
    return None

# ---------- Simple domain detection ----------
def count_keyword_hits(text: str, keywords: set) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)

def guess_domain(filename: str, content: str) -> str:
    name = filename.lower()
    if "salary" in name or "pay" in name or "wage" in name:
        return "salary"
    if "insurance" in name or "policy" in name:
        return "insurance"
    s_hits = count_keyword_hits(content, SALARY_KEYWORDS)
    i_hits = count_keyword_hits(content, INSURANCE_KEYWORDS)
    if s_hits > i_hits:
        return "salary"
    if i_hits > s_hits:
        return "insurance"
    return "salary"  # default

# ---------- Summarization ----------
SUMMARY_TMPL = PromptTemplate(
    template=(
        "You are a concise summarizer. Summarize the text below in 3-6 short bullet points focusing on "
        "practical facts, formulas and items.\n\nText:\n{content}\n\nSummary:"
    ),
    input_variables=["content"],
)

def summarize_text(content: str) -> str:
    llm = get_llm(temperature=0)
    if llm:
        try:
            prompt = SUMMARY_TMPL.format(content=content[:4000])
            resp = llm.invoke(prompt)
            return getattr(resp, "content", str(resp)).strip()
        except Exception:
            pass
    preview = content.strip().replace("\n", " ")
    return textwrap.shorten(preview, width=400, placeholder="...")

# ---------- Build domain vectorstores ----------
def build_domain_vectorstores(files: List[Tuple[str, str]]) -> Tuple[DomainStores, Dict[str, str]]:
    """
    files: list of (filename, content)
    returns (DomainStores, summaries_by_filename)
    """
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ".", " "]
    )

    salary_docs = []
    insurance_docs = []
    summaries: Dict[str, str] = {}

    for fname, content in files:
        domain = guess_domain(fname, content)
        chunks = splitter.split_text(content)
        docs = [Document(page_content=c, metadata={"source": fname, "domain": domain}) for c in chunks]
        if domain == "salary":
            salary_docs.extend(docs)
        else:
            insurance_docs.extend(docs)
        summaries[fname] = summarize_text(content)

    salary_store = FAISS.from_documents(salary_docs, embeddings) if salary_docs else None
    insurance_store = FAISS.from_documents(insurance_docs, embeddings) if insurance_docs else None

    return DomainStores(salary=salary_store, insurance=insurance_store), summaries

# ---------- Prompt templates for agents ----------
AGENT_TMPL = PromptTemplate(
    input_variables=["domain_name", "domain_name_lower", "question", "context"],
    template=(
        "You are the {domain_name} Agent. Answer ONLY {domain_name_lower}-related questions. "
        "If the user's question is not about {domain_name_lower}, reply with: "
        "\"I can only answer {domain_name_lower} questions.\" \n\n"
        "Use the provided context excerpts to answer precisely. Cite sources when helpful. If the context "
        "doesn't contain the answer, say you don't have that information.\n\n"
        "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
    )
)

# ---------- Retrieval + agent answer ----------
def retrieve_context(store: Optional[FAISS], query: str, k: int = 4) -> str:
    if store is None:
        return ""
    try:
        docs = store.similarity_search(query, k=k)
    except Exception:
        docs = []
    snippets = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        snippets.append(f"[Source: {src}]\n{d.page_content}")
    return "\n\n".join(snippets)

def agent_answer(domain: str, stores: DomainStores, question: str) -> str:
    store = stores.salary if domain == "salary" else stores.insurance
    context = retrieve_context(store, question, k=5)
    if not context.strip():
        return f"I couldn't find any {domain} information. Please add relevant {domain} documents in code."
    llm = get_llm(temperature=0)
    prompt = AGENT_TMPL.format(
        domain_name="Salary" if domain == "salary" else "Insurance",
        domain_name_lower=domain,
        question=question,
        context=context
    )
    if llm:
        try:
            resp = llm.invoke(prompt)
            return getattr(resp, "content", str(resp)).strip()
        except Exception:
            return "(LLM failed) Here are the most relevant excerpts:\n\n" + context
    return "(No LLM available) Relevant excerpts:\n\n" + context

# ---------- Coordinator ----------
COORDINATOR_TMPL = PromptTemplate(
    input_variables=["question"],
    template=(
        "Classify the following user question strictly into one of two domains: 'salary' or 'insurance'. "
        "Return exactly one word: salary OR insurance.\n\nQuestion: {question}"
    )
)

def route_question(question: str) -> str:
    llm = get_llm(temperature=0)
    if llm:
        try:
            resp = llm.invoke(COORDINATOR_TMPL.format(question=question))
            lbl = getattr(resp, "content", str(resp)).lower()
            if "salary" in lbl and "insurance" not in lbl:
                return "salary"
            if "insurance" in lbl and "salary" not in lbl:
                return "insurance"
        except Exception:
            pass
    s_hits = count_keyword_hits(question, SALARY_KEYWORDS)
    i_hits = count_keyword_hits(question, INSURANCE_KEYWORDS)
    return "salary" if s_hits >= i_hits else "insurance"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Multi-Agent RAG (Salary & Insurance)", page_icon="🤝", layout="wide")
st.title("🤝 Multi-Agent RAG — Salary & Insurance ")

# session state
if "stores" not in st.session_state:
    st.session_state.stores = DomainStores()
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta}
if "ready" not in st.session_state:
    st.session_state.ready = False

# Build vector stores automatically on first load
if not st.session_state.ready:
    with st.spinner("Preparing built-in Salary & Insurance knowledge…"):
        st.session_state.stores, st.session_state.summaries = build_domain_vectorstores(BUILTIN_FILES)
        st.session_state.ready = True
    st.success("Knowledge loaded. You can start chatting below.")

# Show summaries
if st.session_state.summaries:
    st.subheader("File Summaries")
    for fname, summ in st.session_state.summaries.items():
        with st.expander(fname):
            st.write(summ)

st.divider()

# Display conversation
for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    meta = msg.get("meta", {})
    with st.chat_message(role):
        if meta.get("route"):
            badge = "🧭 Coordinator → " + ("💼 Salary" if meta["route"] == "salary" else "🛡️ Insurance")
            st.markdown(f"<small>{badge}</small>", unsafe_allow_html=True)
        st.write(msg["content"])

# Chat input
user_query = st.chat_input("Ask about salary or insurance…")

if user_query:
    # push user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    # route the query
    route = route_question(user_query)
    # answer via agent
    answer = agent_answer(route, st.session_state.stores, user_query) if st.session_state.ready else "Initializing…"
    # record assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": {"route": route}})
    # render
    with st.chat_message("assistant"):
        st.markdown(f"<small>🧭 Coordinator → {'💼 Salary' if route=='salary' else '🛡️ Insurance'}</small>", unsafe_allow_html=True)
        st.write(answer)

st.caption("Tip: The app uses built-in Salary.txt and Insurance.txt content. Adjust chunking or models in code if needed.")
