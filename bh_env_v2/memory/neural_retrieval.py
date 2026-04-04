"""
Neural Memory Retrieval — Upgrade 3 (DEFERRED -> IMPLEMENTED).

Replaces TF-IDF with dense embedding retrieval for episodic memory.
Falls back gracefully to TF-IDF when neural dependencies are not available.

Supports:
  - sentence-transformers (all-MiniLM-L6-v2, etc.)
  - Custom embedding functions
  - Hybrid mode (neural + TF-IDF score blending)
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..engine.types import Event
from .memory_system import EpisodicMemory, MemorySystem, _tokenize, _compute_tf


# ═════════════════════════════════════════════════════════════════════════════
#  Embedding Interface
# ═════════════════════════════════════════════════════════════════════════════

class EmbeddingModel:
    """Abstract interface for text embedding models."""

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into dense vectors."""
        raise NotImplementedError

    def encode_single(self, text: str) -> List[float]:
        """Encode a single text."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        raise NotImplementedError


class SentenceTransformerModel(EmbeddingModel):
    """Wrapper around sentence-transformers for dense embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dim: Optional[int] = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                # Get dimension from a test encoding
                test = self._model.encode(["test"])
                self._dim = len(test[0])
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load_model()
        return self._dim or 384


class NumpyEmbeddingModel(EmbeddingModel):
    """
    Lightweight embedding model using random projections (NumPy only).
    Not semantically meaningful, but provides a dense vector baseline
    without requiring sentence-transformers or PyTorch.
    """

    def __init__(self, vocab_size: int = 10000, dim: int = 128, seed: int = 42) -> None:
        import numpy as np
        self._dim = dim
        self._rng = np.random.RandomState(seed)
        self._projection = self._rng.randn(vocab_size, dim).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True)
        self._word_to_idx: Dict[str, int] = {}
        self._next_idx: int = 0
        self._vocab_size = vocab_size

    def _get_word_idx(self, word: str) -> int:
        if word not in self._word_to_idx:
            self._word_to_idx[word] = self._next_idx % self._vocab_size
            self._next_idx += 1
        return self._word_to_idx[word]

    def encode(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        results = []
        for text in texts:
            tokens = text.lower().split()
            if not tokens:
                results.append([0.0] * self._dim)
                continue
            indices = [self._get_word_idx(t) for t in tokens]
            vecs = self._projection[indices]
            mean_vec = vecs.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec /= norm
            results.append(mean_vec.tolist())
        return results

    @property
    def dimension(self) -> int:
        return self._dim


# ═════════════════════════════════════════════════════════════════════════════
#  Neural Episodic Memory
# ═════════════════════════════════════════════════════════════════════════════

def _cosine_sim_dense(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class NeuralEpisodicMemory:
    """
    Dense embedding-based episodic memory.
    Stores events with their embeddings for semantic similarity retrieval.
    """

    MAX_EVENTS = 2000

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._model = embedding_model
        self._events: List[Event] = []
        self._embeddings: List[List[float]] = []
        self._importances: List[float] = []

    @staticmethod
    def _importance_score(event: Event) -> float:
        score = event.importance
        tags_lower = [t.lower() for t in event.tags]
        if "shock" in tags_lower:
            score += 10.0
        if "milestone" in tags_lower:
            score += 8.0
        if event.reward > 5.0:
            score += 5.0
        return score

    def add(self, event: Event) -> None:
        """Add an event with its dense embedding."""
        embedding = self._model.encode_single(event.text)
        self._events.append(event)
        self._embeddings.append(embedding)
        self._importances.append(self._importance_score(event))

        if len(self._events) > self.MAX_EVENTS:
            self._events.pop(0)
            self._embeddings.pop(0)
            self._importances.pop(0)

    def retrieve_relevant(self, query: str, k: int = 5) -> List[Tuple[Event, float]]:
        """Retrieve top-k events by dense cosine similarity + importance."""
        if not self._events:
            return []

        query_emb = self._model.encode_single(query)

        scored: List[Tuple[int, float]] = []
        for i, emb in enumerate(self._embeddings):
            sim = _cosine_sim_dense(query_emb, emb)
            combined = sim + self._importances[i] * 0.01
            scored.append((i, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self._events[idx], score) for idx, score in scored[:k]]

    @property
    def size(self) -> int:
        return len(self._events)

    def events_in_range(self, start_step: int, end_step: int) -> List[Event]:
        return [e for e in self._events if start_step <= e.step <= end_step]


# ═════════════════════════════════════════════════════════════════════════════
#  Hybrid Memory — Neural + TF-IDF Blending
# ═════════════════════════════════════════════════════════════════════════════

class HybridEpisodicMemory:
    """
    Blends neural and TF-IDF retrieval scores.
    Falls back to TF-IDF only when neural model is unavailable.

    Score = alpha * neural_score + (1 - alpha) * tfidf_score
    """

    MAX_EVENTS = 2000

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        alpha: float = 0.6,
    ) -> None:
        self._alpha = alpha
        self._tfidf = EpisodicMemory()
        self._neural: Optional[NeuralEpisodicMemory] = None

        if embedding_model is not None:
            self._neural = NeuralEpisodicMemory(embedding_model)
        else:
            # Try to create a lightweight model
            try:
                self._neural = NeuralEpisodicMemory(NumpyEmbeddingModel())
            except ImportError:
                pass  # Pure TF-IDF fallback

    def add(self, event: Event) -> None:
        self._tfidf.add(event)
        if self._neural is not None:
            self._neural.add(event)

    def retrieve_relevant(self, query: str, k: int = 5) -> List[Tuple[Event, float]]:
        """Retrieve with blended scoring."""
        tfidf_results = self._tfidf.retrieve_relevant(query, k=k * 2)

        if self._neural is None:
            return tfidf_results[:k]

        neural_results = self._neural.retrieve_relevant(query, k=k * 2)

        # Build score maps keyed by (step, text) for dedup
        def _key(event: Event) -> Tuple[int, str]:
            return (event.step, event.text[:50])

        tfidf_scores: Dict[Tuple[int, str], Tuple[Event, float]] = {
            _key(e): (e, s) for e, s in tfidf_results
        }
        neural_scores: Dict[Tuple[int, str], Tuple[Event, float]] = {
            _key(e): (e, s) for e, s in neural_results
        }

        # Merge and blend
        all_keys = set(tfidf_scores.keys()) | set(neural_scores.keys())
        blended: List[Tuple[Event, float]] = []

        for key in all_keys:
            t_event, t_score = tfidf_scores.get(key, (None, 0.0))
            n_event, n_score = neural_scores.get(key, (None, 0.0))
            event = t_event or n_event
            if event is None:
                continue
            score = self._alpha * n_score + (1.0 - self._alpha) * t_score
            blended.append((event, score))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:k]

    @property
    def size(self) -> int:
        return self._tfidf.size

    def events_in_range(self, start_step: int, end_step: int) -> List[Event]:
        return self._tfidf.events_in_range(start_step, end_step)


# ═════════════════════════════════════════════════════════════════════════════
#  Enhanced Memory System — Drop-in replacement with neural retrieval
# ═════════════════════════════════════════════════════════════════════════════

class NeuralMemorySystem(MemorySystem):
    """
    Extended MemorySystem that uses HybridEpisodicMemory for retrieval.
    Drop-in replacement for MemorySystem — all compression and context
    assembly logic is inherited.
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        alpha: float = 0.6,
    ) -> None:
        super().__init__()
        # Replace the default episodic memory with hybrid
        self._hybrid_episodic = HybridEpisodicMemory(
            embedding_model=embedding_model,
            alpha=alpha,
        )

    def record_event(self, event: Event) -> None:
        """Override to use hybrid episodic memory."""
        self.working.add(event)
        self.episodic.add(event)
        self._hybrid_episodic.add(event)
        self._all_events_buffer.append(event)
        self.trigger_compression(event.step)

    def context_for_agent(self, step: int, query: str = "") -> str:
        """Override to use hybrid retrieval when query is provided."""
        parts: List[str] = []

        # Strategic insights
        if self.strategic_insights:
            parts.append("=== Strategic Insights ===")
            for si in self.strategic_insights[-3:]:
                risk_str = ", ".join(
                    f"{k}({v})" for k, v in si.recurring_risks.items() if v > 0
                )
                parts.append(
                    f"[Step {si.step}] Trend: {si.trend_direction} | "
                    f"Risks: {risk_str or 'none'} | "
                    f"Reward: {si.net_reward:+.1f} | "
                    f"Rec: {si.recommendation}"
                )

        # Milestone summaries
        if self.milestone_summaries:
            parts.append("=== Milestone Summaries ===")
            for ms in self.milestone_summaries[-5:]:
                outcomes_str = "; ".join(ms.outcomes[:3]) if ms.outcomes else "none"
                risks_str = "; ".join(ms.risks[:3]) if ms.risks else "none"
                parts.append(
                    f"[Phase {ms.phase_from}->{ms.phase_to} @ step {ms.step}] "
                    f"Outcomes: {outcomes_str} | Risks: {risks_str} | "
                    f"Reward: {ms.net_reward:+.1f}"
                )

        # Daily summaries
        if self.daily_summaries:
            parts.append("=== Daily Summaries ===")
            for ds in self.daily_summaries[-5:]:
                top_str = "; ".join(ds.top_events[:3]) if ds.top_events else "none"
                flags = []
                if ds.shock_occurred:
                    flags.append("SHOCK")
                if ds.milestone_occurred:
                    flags.append("MILESTONE")
                flag_str = f" [{','.join(flags)}]" if flags else ""
                parts.append(
                    f"[Steps {ds.step_start}-{ds.step_end}]{flag_str} "
                    f"Top: {top_str} | Reward: {ds.net_reward:+.1f} | "
                    f"Events: {ds.event_count}"
                )

        # Hybrid retrieval (neural + TF-IDF)
        if query:
            relevant = self._hybrid_episodic.retrieve_relevant(query, k=5)
            if relevant:
                parts.append("=== Relevant Past Events (Hybrid Retrieval) ===")
                for ev, score in relevant:
                    parts.append(f"[Step {ev.step}] ({score:.2f}) {ev.text}")

        # Working memory
        working_events = self.working.get_all()
        if working_events:
            parts.append("=== Recent Events (Working Memory) ===")
            for ev in working_events:
                tag_str = f" [{', '.join(ev.tags)}]" if ev.tags else ""
                parts.append(
                    f"[Step {ev.step}]{tag_str} {ev.text} (r={ev.reward:+.1f})"
                )

        return "\n".join(parts)
