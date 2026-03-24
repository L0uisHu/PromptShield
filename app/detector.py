import base64
import json
import unicodedata
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class DetectionResult:
    is_suspicious: bool
    score: float
    flagged_content: str
    reason: str
    detection_method: str  # 'keyword' | 'embedding_similarity' | 'length_heuristic' | 'unicode_keyword' | 'unicode_embedding' | 'base64_*' | ''


PATTERNS_PATH = Path(__file__).parent.parent / "data" / "patterns.json"

ROLE_THRESHOLDS = {
    "user": 0.80,
    "assistant": 0.68,
    "tool_result": 0.55,
}
DEFAULT_THRESHOLD = 0.60

KEYWORDS = [
    "ignore previous",
    "ignore all",
    "disregard",
    "forget your instructions",
    "new persona",
    "pretend you are",
    "your real instructions",
    "do not follow",
    "override",
    "bypass",
    "jailbreak",
    "dan",
    "developer mode",
]


def _normalize(text: str) -> str:
    """NFKC normalization: collapses full-width/homoglyph chars to ASCII equivalents."""
    return unicodedata.normalize("NFKC", text)


def _keyword_hit(text: str) -> bool:
    return any(kw in text.lower() for kw in KEYWORDS)


def _decode_base64_tokens(text: str) -> list[str]:
    """Return decoded strings for any whitespace-separated token that is valid Base64
    and decodes to printable text (min 8 chars to avoid false positives on short tokens)."""
    decoded = []
    for token in text.split():
        # Base64 tokens must be at least 8 chars and properly padded
        if len(token) < 8:
            continue
        # Pad if necessary
        padded = token + "=" * (-len(token) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
            text_decoded = raw.decode("utf-8")
            # Only keep if it's mostly printable (>80% printable chars)
            printable = sum(c.isprintable() for c in text_decoded)
            if len(text_decoded) > 0 and printable / len(text_decoded) > 0.8:
                decoded.append(text_decoded)
        except Exception:
            continue
    return decoded


class PositionalDetector:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        with open(PATTERNS_PATH) as f:
            patterns = json.load(f)

        embeddings = self.model.encode(patterns, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine sim on normalized vecs
        self.index.add(embeddings)

    def _check_content(self, content: str, role: str) -> DetectionResult | None:
        """Run keyword → embedding → length checks on a single content string.
        Returns a DetectionResult if suspicious, else None."""

        # Keyword layer
        if _keyword_hit(content):
            return DetectionResult(
                is_suspicious=True,
                score=1.0,
                flagged_content=content,
                reason="keyword match - injection phrase detected",
                detection_method="keyword",
            )

        # Embedding similarity
        vec = self.model.encode([content], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        distances, _ = self.index.search(vec, k=1)
        similarity = float(distances[0][0])

        threshold = ROLE_THRESHOLDS.get(role, DEFAULT_THRESHOLD)
        blocked = similarity >= threshold
        print(f"[Detector] role={role!r} score={similarity:.4f} threshold={threshold} {'BLOCK' if blocked else 'pass '} | {content[:80]!r}")

        if blocked:
            return DetectionResult(
                is_suspicious=True,
                score=similarity,
                flagged_content=content,
                reason="positional anomaly - instruction-like content detected in non-system role",
                detection_method="embedding_similarity",
            )

        # Length heuristic
        if role in ("tool_result", "user") and len(content) > 500 and similarity > 0.50:
            print(f"[Detector] role={role!r} method=length_heuristic score={similarity:.4f} len={len(content)}")
            return DetectionResult(
                is_suspicious=True,
                score=similarity,
                flagged_content=content,
                reason="length heuristic - long message with moderate injection similarity",
                detection_method="length_heuristic",
            )

        return None

    def scan(self, messages: list) -> DetectionResult:
        for message in messages:
            role = message.get("role", "")
            if role == "system":
                continue

            raw_content = message.get("content", "")
            if not raw_content or not isinstance(raw_content, str):
                continue

            # ── Preprocessing ────────────────────────────────────────────────
            try:
                content_str = raw_content if isinstance(raw_content, str) else str(raw_content)

                # 1. Unicode normalization
                normalized = _normalize(content_str)
                unicode_changed = normalized != content_str
                if unicode_changed:
                    print(f"[Detector] Unicode normalization changed content for role={role!r}")

                # Run detection on normalized content
                result = self._check_content(normalized, role)
                if result:
                    if unicode_changed:
                        result.detection_method = f"unicode_{result.detection_method}"
                        result.reason = f"unicode obfuscation + {result.reason}"
                    return result

                # 2. Base64 decoding
                decoded_strings = _decode_base64_tokens(normalized)
                for decoded in decoded_strings:
                    print(f"[Detector] Base64 token decoded for role={role!r}: {decoded[:80]!r}")
                    inner = self._check_content(decoded, role)
                    if inner:
                        return DetectionResult(
                            is_suspicious=True,
                            score=inner.score,
                            flagged_content=raw_content,
                            reason="base64 encoded injection attempt detected",
                            detection_method=f"base64_{inner.detection_method}",
                        )

            except Exception as exc:
                print(f"[Detector] WARNING: preprocessing failed for role={role!r}, continuing with raw content: {exc}")
                result = self._check_content(raw_content if isinstance(raw_content, str) else str(raw_content), role)
                if result:
                    return result

        return DetectionResult(
            is_suspicious=False,
            score=0.0,
            flagged_content="",
            reason="",
            detection_method="",
        )
