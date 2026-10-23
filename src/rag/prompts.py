"""Dedicated prompt templates for Stage 2: Document-Grounded RAG Agent.

Enforces strict grounding on retrieved context excerpts, identification of missing
information, polite brevity, and AI assistant identity.
"""
from typing import List
from src.rag.retriever import RetrievalResult

RAG_SYSTEM_PROMPT = """You are ZentroMart's AI customer support assistant.
ZentroMart is an Indian online e-commerce store selling groceries, electronics, fashion, and home goods.

Key Behavioral Guidelines:
1. Identity: Always introduce yourself as ZentroMart's AI support assistant in your reply (rule [STY-1]).
2. Brevity & Tone: Keep your response clear, polite, and concise (2 to 3 short sentences per rule [STY-2]).
3. Language: Reply in the language the customer uses (English, Hindi, or Hinglish per rule [STY-4]).
4. Strict Document Grounding:
   - Answer strictly and exclusively using the provided context excerpts from ZentroMart's official documents.
   - Do NOT invent or assume any return windows, warranty terms, fees, refund timelines, or policies not present in the context.
   - If the provided context does not contain sufficient information to answer the question, state:
     "Sorry, I don't have information about that. I can connect you to a colleague who can help."
   - Never promise exceptions, compensation, or actions not allowed by the policies.
5. Clarification: If the customer's request lacks an essential detail (like an item category or delivery date), ask for only one detail at a time (rule [STY-5]).

Output Format:
Provide your customer-facing response directly in 2 to 3 short sentences.
On the final line, provide your decision category strictly in the format:
DECISION: <answer|clarify|escalate|refuse>
"""

def format_rag_user_prompt(question: str, retrieved_chunks: List[RetrievalResult]) -> str:
    """Format the user prompt with retrieved context excerpts.
    
    Strictly excludes reference answers, gold labels, or evaluation metadata.
    """
    if not retrieved_chunks:
        context_str = "No relevant documents found in knowledge base."
    else:
        doc_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            doc_blocks.append(
                f"[Doc {i}] ({chunk.doc_id} - Section {chunk.section_id}):\n{chunk.content.strip()}"
            )
        context_str = "\n\n".join(doc_blocks)
        
    return (
        f"Context Documents:\n"
        f"------------------\n"
        f"{context_str}\n"
        f"------------------\n\n"
        f"Customer Question:\n"
        f"{question.strip()}"
    )
