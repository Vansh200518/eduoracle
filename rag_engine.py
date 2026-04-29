import os
import cohere
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_KEY,
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    return qa_chain

def ask_question(chain, question):
    result = chain({"question": question})
    answer = result["answer"]
    sources = list(set([doc.metadata["source"] for doc in result["source_documents"]]))
    return answer, sources
