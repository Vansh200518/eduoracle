import cohere
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

GROQ_KEY = "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"
COHERE_KEY = "64H2U9s9f2E0w1sqXrmitCQLGi9WliB4LsDEukz6"

co = cohere.Client(COHERE_KEY)

class CohereEmbeddings:
    def embed_documents(self, texts):
        results = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = [t[:500] for t in texts[i:i+batch_size]]
            response = co.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            results.extend(response.embeddings)
        return results

    def embed_query(self, text):
        response = co.embed(
            texts=[text[:500]],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]

def load_rag_chain():
    embeddings = CohereEmbeddings()
    vectordb = FAISS.load_local(
        "./faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_KEY,
        temperature=0.3
    )
    return retriever, llm

chat_history = []

def ask_question(chain, question):
    retriever, llm = chain
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata["source"] for d in docs]))

    messages = []
    for h in chat_history[-6:]:
        messages.append(HumanMessage(content=h[0]))
        messages.append(AIMessage(content=h[1]))
    messages.append(HumanMessage(content=f"""You are EduOracle, an AI study assistant for Class 9-12 students.
Answer ONLY based on the textbook context below. If answer not in context, say so.

Context:
{context}

Question: {question}"""))

    response = llm.invoke(messages)
    answer = response.content
    chat_history.append((question, answer))
    return answer, sources
