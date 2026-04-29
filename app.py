import streamlit as st
from rag_engine import load_rag_chain, ask_question

st.set_page_config(
    page_title="EduOracle — AI Study Assistant",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def get_chain():
    return load_rag_chain()

st.title("🎓 EduOracle")
st.markdown("**AI-Powered Study Assistant | Class 9–12 | All Subjects**")
st.divider()

with st.sidebar:
    st.header("📚 About EduOracle")
    st.info("""
    **Knowledge Base:**
    - Class 9 & 10 NCERT
    - Class 11 & 12 NCERT
    - All subjects loaded
    
    **Powered by:**
    - Groq AI (Llama 3)
    - LangChain RAG
    - FAISS Vector Store
    - Cohere Embeddings
    """)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("*Answers grounded in textbooks. Sources shown for every answer.*")

    st.markdown("**💡 Try these questions:**")
    sample_questions = [
        "Explain Newton's laws of motion",
        "What is photosynthesis?",
        "Explain the French Revolution",
        "What are types of chemical bonds?",
        "Solve quadratic equations"
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.demo_question = q

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("Loading AI brain... (~30 seconds)"):
        st.session_state.chain = get_chain()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📖 Sources: {', '.join(message['sources'])}")

if prompt := st.chat_input("Ask anything from your textbooks..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching textbooks..."):
            answer, sources = ask_question(st.session_state.chain, prompt)
        st.write(answer)
        if sources:
            st.caption(f"📖 Sources: {', '.join(sources)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
