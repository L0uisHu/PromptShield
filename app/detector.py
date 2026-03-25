import base64
import codecs
import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class DetectionResult:
    is_suspicious: bool
    score: float
    flagged_content: str
    reason: str
    detection_method: str  # see layer constants below


PATTERNS_PATH = Path(__file__).parent.parent / "data" / "patterns.json"

ROLE_THRESHOLDS = {
    "user": 0.80,
    "assistant": 0.68,
    "tool_result": 0.55,
}
DEFAULT_THRESHOLD = 0.60

# ── Layer 5: Keyword matching ─────────────────────────────────────────────────
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

# ── Layer 2: Invisible / control character injection ─────────────────────────
INVISIBLE_CHAR_PATTERN = re.compile(
    r'[\u00ad'           # SOFT HYPHEN
    r'\u200b-\u200f'     # zero-width space, ZWNJ, ZWJ, LRM, RLM
    r'\u2028\u2029'      # LINE SEPARATOR, PARAGRAPH SEPARATOR
    r'\u202a-\u202e'     # LRE, RLE, PDF, LRO, RLO (includes RTL override U+202E)
    r'\u2060-\u2064'     # WORD JOINER, invisible math operators
    r'\u206a-\u206f'     # deprecated formatting chars
    r'\ufeff'            # ZERO WIDTH NO-BREAK SPACE / BOM
    r']'
)
INVISIBLE_CHAR_THRESHOLD = 3

# ── Layer 3: Prompt delimiter injection ──────────────────────────────────────
# Tier A: delimiter + role context → hard block (score 1.0) for any role
DELIMITER_HARD_PATTERN = re.compile(
    r'('
    r'<\|im_start\|>\s*(?:system|assistant)'    # ChatML role injection
    r'|\[INST\]\s*<<SYS>>'                       # LLaMA-2 system block
    r'|<<SYS>>'                                  # LLaMA-2 system wrap
    r'|###\s*(?:System|Instruction)[\s:#]'       # Alpaca/Vicuna system headers
    r'|<\|system\|>'                             # Phi-3/Zephyr system
    r'|<\|assistant\|>'                          # Phi-3/Zephyr assistant injection
    r'|<start_of_turn>\s*model'                  # Gemma model role injection
    r'|<\/?(?:system|instructions?)>'           # Generic XML role semantics
    r')',
    re.IGNORECASE
)

# Tier B: standalone delimiter token → soft block only for tool_result (score 0.70)
DELIMITER_SOFT_PATTERN = re.compile(
    r'('
    r'<\|im_start\|>'
    r'|<\|im_end\|>'
    r'|<\|endoftext\|>'
    r'|\[/?INST\]'
    r'|<</SYS>>'
    r'|<\|end\|>'
    r'|<\|EOT\|>'
    r'|<start_of_turn>'
    r'|<end_of_turn>'
    r'|<\|user\|>'
    r')',
    re.IGNORECASE
)

# ── Layer 4: Alternative encoding detection ───────────────────────────────────
# Require ≥4 consecutive encoded chars to avoid false positives on %20, &amp; etc.
HEX_ESCAPE_PATTERN = re.compile(r'(?:\\x[0-9a-fA-F]{2}){4,}')
URL_ENCODE_PATTERN = re.compile(r'(?:%[0-9a-fA-F]{2}){4,}')
HTML_ENTITY_PATTERN = re.compile(r'(?:&#\d{2,3};){4,}|(?:&#x[0-9a-fA-F]{2};){4,}')

# ── Layer 6: Fuzzy keyword matching ──────────────────────────────────────────
FUZZY_KEYWORDS = [
    "ignore previous", "ignore all", "disregard", "forget your instructions",
    "new persona", "pretend you are", "your real instructions", "do not follow",
    "override", "bypass", "jailbreak", "developer mode",
    "forget everything", "new instructions", "act as", "roleplay as",
    "no restrictions", "without restrictions", "reveal instructions",
    "print instructions",
]

# ── Layer 7: Exfiltration pattern detection ───────────────────────────────────
EXFILTRATION_PATTERN = re.compile(
    r'('
    r'repeat\s+(?:everything|all|the\s+(?:text|words|content|instructions?))?\s*(?:above|before|prior)'
    r'|(?:print|output|show|display|reveal|return|give\s+me|tell\s+me|share)\s+'
        r'(?:your\s+)?(?:system\s+prompt|initial\s+(?:prompt|instructions?)|'
        r'(?:all\s+)?(?:hidden\s+)?instructions?|context\s+window|'
        r'everything\s+(?:above|before)|(?:full\s+)?configuration|'
        r'(?:original\s+)?(?:prompt|directives?|guidelines?))'
    r'|what\s+(?:are|were|is)\s+(?:your\s+)?(?:initial\s+|original\s+|full\s+)?'
        r'(?:instructions?|prompt|directives?|guidelines?|system\s+(?:prompt|message))'
    r'|summarize\s+(?:your\s+)?(?:system|instructions?|context|prompt)'
    r'|encode\s+(?:your\s+)?(?:system\s+)?(?:instructions?|prompt)\s+(?:in\s+)?(?:base64|hex|json)'
    r'|(?:exfiltrate|leak|extract)\s+(?:the\s+)?(?:data|prompt|instructions?|context)'
    r'|beginning\s+of\s+(?:this\s+)?(?:conversation|context|prompt|session)'
    r')',
    re.IGNORECASE
)

# ── Layer 8: Instruction density scoring ─────────────────────────────────────
IMPERATIVE_VERBS = {
    "ignore", "forget", "disregard", "override", "bypass", "reveal", "output",
    "print", "say", "tell", "pretend", "respond", "act", "roleplay", "comply",
    "obey", "follow", "execute", "perform", "show", "display", "return",
    "enable", "disable", "unlock", "switch", "reset", "delete", "remove",
    "expose", "leak", "exfiltrate", "dump", "extract",
}
DENSITY_THRESHOLD = 0.08    # 8% of tokens are imperative verbs
DENSITY_MIN_TOKENS = 30     # only apply to messages with ≥30 tokens
DENSITY_MIN_VERB_COUNT = 4  # require ≥4 hits to avoid short message false positives

# ── Layer 9: Repetition / flooding detection ──────────────────────────────────
FLOOD_WINDOW = 50
FLOOD_REPEAT_RATIO = 0.60         # one token >60% of a window
FLOOD_MIN_TOKENS = 20
FLOOD_PHRASE_REPEAT_THRESHOLD = 8  # bigram repeated ≥8 times


# ─────────────────────────────────────────────────────────────────────────────
# Layer functions
# ─────────────────────────────────────────────────────────────────────────────

def _check_invisible_chars(text: str) -> DetectionResult | None:
    """Layer 2: detect invisible/control chars used to split keywords."""
    hits = INVISIBLE_CHAR_PATTERN.findall(text)
    if len(hits) >= INVISIBLE_CHAR_THRESHOLD:
        return DetectionResult(
            is_suspicious=True,
            score=min(1.0, 0.6 + 0.04 * len(hits)),
            flagged_content=text,
            reason=f"invisible/control character injection: {len(hits)} suspicious codepoints found",
            detection_method="invisible_char",
        )
    return None


def _check_prompt_delimiters(text: str, role: str) -> DetectionResult | None:
    """Layer 3: detect chat-template boundary tokens used to inject fake system turns."""
    # Tier A: hard block — delimiter + role context
    match = DELIMITER_HARD_PATTERN.search(text)
    if match:
        return DetectionResult(
            is_suspicious=True,
            score=1.0,
            flagged_content=text,
            reason=f"prompt delimiter injection: role-injecting token '{match.group(0).strip()}'",
            detection_method="prompt_delimiter",
        )
    # Tier B: standalone delimiter — only block for untrusted tool_result
    if role == "tool_result":
        match = DELIMITER_SOFT_PATTERN.search(text)
        if match:
            return DetectionResult(
                is_suspicious=True,
                score=0.70,
                flagged_content=text,
                reason=f"prompt delimiter in untrusted tool result: '{match.group(0).strip()}'",
                detection_method="prompt_delimiter",
            )
    return None


def _decode_alt_encodings(text: str) -> list[tuple[str, str]]:
    """Layer 4: decode hex escapes, URL encoding, HTML entities, ROT13."""
    results = []

    # 1. Hex escape sequences: \x69\x67\x6e...
    for match in HEX_ESCAPE_PATTERN.finditer(text):
        try:
            hex_str = match.group(0).replace('\\x', '')
            decoded = bytes.fromhex(hex_str).decode('utf-8', errors='ignore')
            if decoded.strip():
                results.append(('hex_escape', decoded))
        except Exception:
            pass

    # 2. URL percent-encoding: %69%67%6e...
    for match in URL_ENCODE_PATTERN.finditer(text):
        try:
            decoded = unquote(match.group(0))
            if decoded != match.group(0) and decoded.strip():
                results.append(('url_encoding', decoded))
        except Exception:
            pass

    # 3. HTML entities: &#105;&#103;... (stdlib handles named + numeric)
    if HTML_ENTITY_PATTERN.search(text):
        decoded = html.unescape(text)
        if decoded != text:
            results.append(('html_entity', decoded))

    # 4. ROT13: gated by downstream keyword/embedding check to avoid FPs
    rot13_decoded = codecs.decode(text, 'rot_13')
    results.append(('rot13', rot13_decoded))

    return results


def _check_fuzzy_keywords(text: str) -> DetectionResult | None:
    """Layer 6: Levenshtein sliding window to catch l33tspeak/misspelled keywords."""
    hit, matched_kw = _fuzzy_keyword_hit(text)
    if hit:
        return DetectionResult(
            is_suspicious=True,
            score=0.90,
            flagged_content=text,
            reason=f"fuzzy keyword match - near-match for '{matched_kw}'",
            detection_method="fuzzy_keyword",
        )
    return None


def _check_exfiltration(text: str) -> DetectionResult | None:
    """Layer 7: detect system-prompt extraction attempts."""
    match = EXFILTRATION_PATTERN.search(text)
    if match:
        return DetectionResult(
            is_suspicious=True,
            score=0.95,
            flagged_content=text,
            reason=f"exfiltration pattern detected: '{match.group(0).strip()[:60]}'",
            detection_method="exfiltration",
        )
    return None


def _check_instruction_density(text: str) -> DetectionResult | None:
    """Layer 8: detect slow-boil attacks via high imperative verb density."""
    tokens = text.lower().split()
    if len(tokens) < DENSITY_MIN_TOKENS:
        return None
    verb_hits = [t.strip('.,!?;:') for t in tokens if t.strip('.,!?;:') in IMPERATIVE_VERBS]
    if len(verb_hits) < DENSITY_MIN_VERB_COUNT:
        return None
    density = len(verb_hits) / len(tokens)
    if density >= DENSITY_THRESHOLD:
        score = min(1.0, 0.70 + (density - DENSITY_THRESHOLD) * 3.0)
        return DetectionResult(
            is_suspicious=True,
            score=round(score, 4),
            flagged_content=text,
            reason=f"instruction density: {density:.1%} of tokens are imperative verbs ({len(verb_hits)} hits)",
            detection_method="instruction_density",
        )
    return None


def _check_repetition_flood(text: str) -> DetectionResult | None:
    """Layer 9: detect context-flooding via token/phrase repetition."""
    tokens = text.lower().split()
    if len(tokens) < FLOOD_MIN_TOKENS:
        return None

    # Check 1: single token dominance in any sliding window
    for start in range(0, len(tokens), FLOOD_WINDOW // 2):
        window = tokens[start:start + FLOOD_WINDOW]
        if len(window) < 10:
            break
        counts = Counter(window)
        most_common_token, most_common_count = counts.most_common(1)[0]
        ratio = most_common_count / len(window)
        if ratio >= FLOOD_REPEAT_RATIO:
            return DetectionResult(
                is_suspicious=True,
                score=min(1.0, 0.65 + ratio * 0.35),
                flagged_content=text,
                reason=f"repetition flood: token '{most_common_token}' is {ratio:.0%} of a {len(window)}-token window",
                detection_method="repetition_flood",
            )

    # Check 2: bigram phrase repetition
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    bigram_counts = Counter(bigrams)
    for phrase, count in bigram_counts.most_common(5):
        if count >= FLOOD_PHRASE_REPEAT_THRESHOLD:
            return DetectionResult(
                is_suspicious=True,
                score=min(1.0, 0.60 + count * 0.03),
                flagged_content=text,
                reason=f"repetition flood: phrase '{phrase}' repeated {count} times",
                detection_method="repetition_flood",
            )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers (shared)
# ─────────────────────────────────────────────────────────────────────────────

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
        if len(token) < 8:
            continue
        padded = token + "=" * (-len(token) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
            text_decoded = raw.decode("utf-8")
            printable = sum(c.isprintable() for c in text_decoded)
            if len(text_decoded) > 0 and printable / len(text_decoded) > 0.8:
                decoded.append(text_decoded)
        except Exception:
            continue
    return decoded


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy matching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Pure Python Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j] + (ca != cb), curr[-1] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def _fuzzy_max_dist(keyword: str) -> int:
    return 1 if len(keyword) <= 8 else 2


def _fuzzy_keyword_hit(text: str) -> tuple[bool, str]:
    lower = text.lower()
    words = lower.split()
    for kw in FUZZY_KEYWORDS:
        kw_words = kw.split()
        kw_len = len(kw_words)
        max_dist = _fuzzy_max_dist(kw)
        for i in range(len(words) - kw_len + 1):
            window = " ".join(words[i:i + kw_len])
            if window == kw:
                continue  # exact match already handled by keyword layer
            if _levenshtein(window, kw) <= max_dist:
                return True, kw
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

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
        """Keyword → embedding similarity → length heuristic on a single content string."""

        # Keyword
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

            try:
                # ── Layer 2: Invisible char check (on raw, before normalization) ──
                result = _check_invisible_chars(raw_content)
                if result:
                    return result

                # ── Layer 3: Prompt delimiter check (on raw) ──────────────────
                result = _check_prompt_delimiters(raw_content, role)
                if result:
                    return result

                # ── Layer 4: Alternative encoding check (on raw) ──────────────
                for encoding_name, decoded_text in _decode_alt_encodings(raw_content):
                    inner = self._check_content(decoded_text, role)
                    if inner:
                        return DetectionResult(
                            is_suspicious=True,
                            score=inner.score,
                            flagged_content=raw_content,
                            reason=f"{encoding_name} encoded injection detected",
                            detection_method=f"alt_encoding_{encoding_name}",
                        )

                # ── Preprocessing: Unicode normalization ──────────────────────
                normalized = _normalize(raw_content)
                unicode_changed = normalized != raw_content
                if unicode_changed:
                    print(f"[Detector] Unicode normalization changed content for role={role!r}")

                # ── Layer 5: Keyword check (on normalized) ────────────────────
                if _keyword_hit(normalized):
                    method = "unicode_keyword" if unicode_changed else "keyword"
                    reason = ("unicode obfuscation + " if unicode_changed else "") + "keyword match - injection phrase detected"
                    return DetectionResult(
                        is_suspicious=True,
                        score=1.0,
                        flagged_content=raw_content,
                        reason=reason,
                        detection_method=method,
                    )

                # ── Layer 6: Fuzzy keyword check (on normalized) ──────────────
                result = _check_fuzzy_keywords(normalized)
                if result:
                    if unicode_changed:
                        result.detection_method = f"unicode_{result.detection_method}"
                    return result

                # ── Layer 7: Exfiltration pattern check (on normalized) ───────
                result = _check_exfiltration(normalized)
                if result:
                    return result

                # ── Layer 8: Instruction density check (on normalized) ────────
                result = _check_instruction_density(normalized)
                if result:
                    return result

                # ── Layer 9: Repetition flood check (on normalized) ───────────
                result = _check_repetition_flood(normalized)
                if result:
                    return result

                # ── Preprocessing: Base64 decoding ────────────────────────────
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

                # ── Layer 10: Embedding similarity + length heuristic ─────────
                result = self._check_content(normalized, role)
                if result:
                    if unicode_changed:
                        result.detection_method = f"unicode_{result.detection_method}"
                        result.reason = f"unicode obfuscation + {result.reason}"
                    return result

            except Exception as exc:
                print(f"[Detector] WARNING: preprocessing failed for role={role!r}: {exc}")
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
