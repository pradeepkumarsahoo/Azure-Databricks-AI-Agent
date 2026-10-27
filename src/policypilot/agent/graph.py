# Author: Pradeep Sahoo
"""LangGraph agent: plan -> retrieve -> rerank -> answer -> verify.

Genuinely agentic rather than a bare RAG chain: the plan step decides what to search for
(and expands the query into multiple variants for better recall), the rerank step re-scores
candidates with a cross-encoder for precision, and the verify step is a hard gate —
an answer without a citation back to a retrieved chunk is replaced with a refusal instead
of being returned to the user.
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from policypilot.agent.llm import LLMClient
from policypilot.agent.tools import filing_metadata, format_context, retrieve
from policypilot.config import DEFAULT_TICKERS, get_settings
from policypilot.retrieval.base import Reranker, SearchResult, VectorStore

CITATION_RE = re.compile(r"\[\d+\]")

PLAN_SYSTEM_PROMPT = (
    "You turn a user's question about SEC 10-K filings into search queries for a semantic "
    "search index over filing text, and detect which company (if any) the question is about. "
    "Known companies (by ticker): " + ", ".join(DEFAULT_TICKERS) + ". "
    "The question may name a company by ticker OR by common name (e.g. 'Apple' means AAPL) "
    "— resolve either form to the ticker symbol.\n\n"
    "Generate 3 diverse search queries that approach the question from different angles to "
    "maximize retrieval recall. Reply with ONLY a JSON object:\n"
    '{"queries": ["<primary query>", "<alternate 1>", "<alternate 2>"], "ticker": "<TICKER or null>"}'
)

ANSWER_SYSTEM_PROMPT = (
    "You are PolicyPilot, a compliance research assistant that answers questions about "
    "SEC 10-K filings using ONLY the numbered context provided below. This is a strict "
    "citation requirement: every sentence containing a factual claim MUST end with a "
    "bracketed citation number like [1] or [2] matching the context item it came from — "
    "no other citation style (no footnotes, no parentheses, no 'Source:' lines). "
    "Example: 'The filing discloses reliance on a limited number of suppliers [1].' "
    "If the context does not contain enough information to answer, say so plainly instead "
    "of guessing — never answer from general knowledge.\n\nContext:\n{context}"
)

REFUSAL_MESSAGE = (
    "I don't have enough grounded information in the ingested filings to answer that "
    "confidently. Try rephrasing, or ask about one of the ingested companies."
)


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    search_queries: list[str]
    ticker_hint: str | None
    results: list[SearchResult]
    metadata: dict | None
    draft_answer: str
    final_answer: str
    grounded: bool


def build_graph(
    llm: LLMClient,
    store: VectorStore,
    reranker: Reranker | None = None,
    settings=None,
):
    _settings = settings or get_settings()

    def plan_node(state: AgentState) -> AgentState:
        reply = llm.complete(
            system=PLAN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["question"]}],
        ).strip()
        try:
            parsed = json.loads(reply)
            # Support both new multi-query format and legacy single-query format
            if "queries" in parsed:
                queries = [str(q).strip() for q in parsed["queries"] if q]
            else:
                sq = str(parsed.get("search_query") or "").strip()
                queries = [sq] if sq else []
            ticker = parsed.get("ticker")
            ticker = str(ticker).strip().upper() if ticker else None
        except (json.JSONDecodeError, AttributeError):
            queries, ticker = [], None
        ticker_hint = ticker if ticker in DEFAULT_TICKERS else None
        primary = queries[0] if queries else state["question"]
        return {
            "search_query": primary,
            "search_queries": queries or [primary],
            "ticker_hint": ticker_hint,
        }

    def retrieve_node(state: AgentState) -> AgentState:
        ticker_hint = state.get("ticker_hint")
        queries = state.get("search_queries") or [state["search_query"]]
        k_per_query = max(3, _settings.top_k_retrieval // len(queries))
        seen_texts: set[str] = set()
        all_results: list[SearchResult] = []
        for q in queries:
            for r in retrieve(store, q, k=k_per_query, ticker=ticker_hint):
                if r.text not in seen_texts:
                    seen_texts.add(r.text)
                    all_results.append(r)
        all_results.sort(key=lambda r: r.score, reverse=True)
        results = all_results[: _settings.top_k_retrieval]
        metadata = filing_metadata(ticker_hint) if ticker_hint else None
        return {"results": results, "metadata": metadata}

    def rerank_node(state: AgentState) -> AgentState:
        results = state.get("results") or []
        if not results or reranker is None:
            return {}
        reranked = reranker.rerank(state["question"], results, top_k=_settings.top_k_rerank)
        return {"results": reranked}

    def answer_node(state: AgentState) -> AgentState:
        context = format_context(state["results"])
        if not context:
            return {"draft_answer": REFUSAL_MESSAGE}
        system = ANSWER_SYSTEM_PROMPT.format(context=context)
        draft = llm.complete(
            system=system, messages=[{"role": "user", "content": state["question"]}]
        )
        return {"draft_answer": draft}

    def verify_node(state: AgentState) -> AgentState:
        draft = state["draft_answer"]
        grounded = bool(state.get("results")) and bool(CITATION_RE.search(draft))
        final = draft if grounded else REFUSAL_MESSAGE
        return {"final_answer": final, "grounded": grounded}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


def ask(
    llm: LLMClient,
    store: VectorStore,
    question: str,
    reranker: Reranker | None = None,
) -> AgentState:
    app = build_graph(llm, store, reranker=reranker)
    return app.invoke({"question": question})
