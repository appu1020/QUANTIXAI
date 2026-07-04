"""
rag/prompt_builder.py — Structured prompt templates for the RAG reasoning layer.

Builds prompts that combine retrieved financial knowledge with the current
prediction context to generate intelligent, grounded explanations.
"""

from __future__ import annotations
from typing import Any
import re

# ── System Prompts ────────────────────────────────────────────────────────────
_BASE_RAG_SYSTEM = """You are QuantixAI, an elite AI financial assistant and technical analyst.

STRICT RAG RULES:
1. Base your financial/technical answers PRIMARILY on the RETRIEVED KNOWLEDGE provided.
2. If the retrieved knowledge doesn't fully answer the question, supplement it using your expert general knowledge, but ALWAYS clarify what was retrieved vs what is your general knowledge.
3. NEVER say "I couldn't find that information" abruptly. Instead say: "I don't have this specific data in my retrieved documents, but generally speaking..."
4. Always be professional, concise, and analytical.
5. If you provide financial advice, add a brief disclaimer."""

_GENERAL_CHAT_SYSTEM = """You are QuantixAI, a friendly, intelligent, and highly capable AI assistant.
You can chat about anything, tell jokes, write code, or just be a helpful companion.
You also specialize in finance, but right now the user is just chatting normally.
Keep responses natural, helpful, and concise. Do NOT sound like a rigid robot.
Remember user details if they share them."""

_LIVE_DATA_SYSTEM = """You are QuantixAI, an expert market data interpreter.
You are being provided with LIVE MARKET DATA. Summarize the price action concisely.
Mention the current price, percentage change, and volume. Highlight if it's a strong up/down move."""

_PREDICTION_SYSTEM = """You are QuantixAI, a quantitative trading AI.
You are being provided with the outputs of a Deep Learning Ensemble model (LSTM, BiLSTM, GRU, CNN).
Explain the prediction (BUY/SELL/HOLD), the confidence score, and the technical indicators supporting it.
Be objective and always mention that AI predictions carry risk."""

# ── Templates ─────────────────────────────────────────────────────────────────
_RAG_TEMPLATE = """\
{system}

{history_section}
=== RETRIEVED FINANCIAL KNOWLEDGE ===
{retrieved_context}

=== USER QUESTION ===
{question}

=== ANSWER ==="""

_GENERAL_TEMPLATE = """\
{system}

{history_section}
=== USER QUESTION ===
{question}

=== ANSWER ==="""

_CONTEXTUAL_TEMPLATE = """\
{system}

{history_section}
=== MARKET/PREDICTION CONTEXT ===
{prediction_context}

=== RETRIEVED KNOWLEDGE (Optional) ===
{retrieved_context}

=== USER QUESTION ===
{question}

=== ANSWER ==="""

def build_prompt(
    intent: str,
    question: str,
    retrieved_docs: list[dict[str, Any]] | None = None,
    history: list[dict[str, str]] | None = None,
    prediction_context: str | None = None,
) -> str:
    """Build dynamic prompt based on intent."""
    history_section = _format_history(history)
    retrieved_context = _format_retrieved_docs(retrieved_docs) if retrieved_docs else "No documents retrieved."

    if intent in ["Greeting", "Conversation", "General Knowledge", "Programming", "AI", "Identity", "Personal Memory"]:
        return _GENERAL_TEMPLATE.format(
            system=_GENERAL_CHAT_SYSTEM,
            history_section=history_section,
            question=question
        )
    elif intent in ["Prediction", "Stock Analysis", "Risk Analysis"]:
        return _CONTEXTUAL_TEMPLATE.format(
            system=_PREDICTION_SYSTEM,
            history_section=history_section,
            prediction_context=prediction_context or "No prediction data provided.",
            retrieved_context=retrieved_context,
            question=question
        )
    elif intent in ["Live Data", "Market Sentiment", "Portfolio"]:
        return _CONTEXTUAL_TEMPLATE.format(
            system=_LIVE_DATA_SYSTEM,
            history_section=history_section,
            prediction_context=prediction_context or "No live data provided.",
            retrieved_context=retrieved_context,
            question=question
        )
    else:
        # Default to RAG for Financial, Indicator Explanation, etc.
        return _RAG_TEMPLATE.format(
            system=_BASE_RAG_SYSTEM,
            history_section=history_section,
            retrieved_context=retrieved_context,
            question=question
        )

def _format_history(history: list[dict[str, str]] | None, max_turns: int = 6) -> str:
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = ["=== CONVERSATION HISTORY ==="]
    for turn in recent:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content[:300]}")
    lines.append("")
    return "\n".join(lines)

def _format_retrieved_docs(docs: list[dict[str, Any]], max_chars_per_doc: int = 1000) -> str:
    if not docs:
        return "No relevant knowledge retrieved."
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.get("source", "unknown")
        title = doc.get("title", "")
        text = doc.get("text", "")[:max_chars_per_doc]
        score = doc.get("cross_score", doc.get("rrf_score", doc.get("score", 0.0)))
        doc_type = doc.get("doc_type", "")
        header = f"[{i}] Source: {source} | Rel: {score:.2f}"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)

def classify_query_intent(question: str) -> str:
    """Classify the user's question into one of the Hybrid intents using heuristics."""
    q_lower = question.lower().strip()
    
    # Greetings & Identity
    if q_lower in ["hello", "hi", "hey", "good morning", "good evening", "how are you", "how are you?"]:
        return "Greeting"
    if any(k in q_lower for k in ["who are you", "what can you do", "your name"]):
        return "Identity"
    
    # Personal Memory
    if any(k in q_lower for k in ["my name is", "i am", "who am i", "do you remember"]):
        return "Personal Memory"
        
    # General Conversation / Humor
    if any(k in q_lower for k in ["tell me a joke", "thank you", "thanks", "bye", "goodbye"]):
        return "Conversation"
        
    # Programming / AI
    if any(k in q_lower for k in ["write code", "python", "javascript", "html", "react", "django", "llm", "rag", "machine learning"]):
        return "Programming"

    # Live Data
    if any(k in q_lower for k in ["current price", "live price", "latest price", "how is the market", "market today"]):
        return "Live Data"

    # Prediction
    if any(k in q_lower for k in ["predict", "will it go up", "will it drop", "buy or sell", "forecast"]):
        return "Prediction"

    # Technical Indicators (RAG)
    if any(k in q_lower for k in ["what is", "define", "explain", "macd", "rsi", "vwap", "bollinger", "adx", "moving average", "ema", "sma", "ichimoku"]):
        return "Indicator Explanation"
        
    # Default to Financial RAG
    return "Financial"
