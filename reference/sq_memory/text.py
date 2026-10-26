"""Tokenising, entity spotting and date parsing — no external dependencies.

Multilingual by construction: Latin-script words are lower-cased and lightly
stemmed; runs of CJK / Hangul characters become overlapping bigrams, which is
the usual trick for languages without spaces.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.+#]*|[぀-ヿ㐀-鿿가-힯]+|[^\W\d_]+", re.UNICODE)
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_STOP = frozenset(
    "a an and are as at be been but by can did do does for from had has have how i if in into is it its "
    "me my of on or our so than that the their them then there these they this to was we were what when "
    "where which who why will with you your about would could should just not no yes".split()
)


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "ly", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "")
    out: list[str] = []
    for m in _WORD.finditer(text):
        tok = m.group(0)
        if _CJK.search(tok):
            if len(tok) == 1:
                out.append(tok)
            else:
                out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
            continue
        low = tok.lower().strip(".-_")
        if not low or low in _STOP:
            continue
        out.append(_stem(low))
    return out


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mentions(text: str, term: str) -> bool:
    """Whole-word, case-insensitive match (substring match for CJK terms)."""
    t = (term or "").lower()
    if not t:
        return False
    low = (text or "").lower()
    if _CJK.search(t):
        return t in low
    return re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low) is not None


_CAPS = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:[\s\-][A-Z0-9][A-Za-z0-9\-]*){0,3})\b")
_CODEISH = re.compile(r"\b([A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+|[A-Z]{2,}\d+[A-Z0-9]*)\b")


def extract_entities(text: str, vocabulary: Optional[Iterable[str]] = None, max_entities: int = 12) -> list[str]:
    """Return canonical entity names found in ``text``.

    ``vocabulary`` (product-supplied, e.g. concept ids, compound names,
    algorithm names) wins; capitalised phrases and code-like tokens such as
    ``ML-KEM-768`` or ``RSA2048`` are the fallback.
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        k = name.strip().lower()
        if k and k not in seen and k not in _STOP:
            seen.add(k)
            found.append(name.strip())

    low = (text or "").lower()
    if vocabulary:
        for term in sorted({v for v in vocabulary if v}, key=len, reverse=True):
            if mentions(low, term):
                add(term)
    for m in _CODEISH.finditer(text or ""):
        add(m.group(1))
    for m in _CAPS.finditer(text or ""):
        phrase = m.group(1)
        # skip a lone capitalised word at the start of a sentence
        start = m.start(1)
        if " " not in phrase and "-" not in phrase and (start == 0 or text[max(0, start - 2):start].strip() in {".", "!", "?", ""}):
            continue
        add(phrase)
    return found[:max_entities]


_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YM = re.compile(r"\b(\d{4})-(\d{2})\b")
_LAST_N = re.compile(r"\b(?:last|past)\s+(\d+)\s+(day|week|month)s?\b", re.I)


def _day(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_time_range(query: str, now: Optional[datetime] = None) -> Optional[tuple[datetime, datetime]]:
    """Find a time window mentioned in a query (en/ko/ja/zh basics)."""
    now = now or datetime.now(timezone.utc)
    q = query or ""
    ql = q.lower()
    m = _ISO.search(q)
    if m:
        d = datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
        return d, d + timedelta(days=1)
    m = _LAST_N.search(q)
    if m:
        n = int(m[1]); unit = m[2].lower()
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        return now - timedelta(days=days), now
    table = [
        (("yesterday", "어제", "昨日", "昨天"), 1, 1),
        (("today", "오늘", "今日", "今天"), 0, 0),
        (("last week", "지난주", "지난 주", "先週", "上周"), 14, 7),
        (("this week", "이번 주", "이번주", "今週", "本周"), 7, 0),
        (("last month", "지난달", "지난 달", "先月", "上个月"), 60, 30),
        (("this month", "이번 달", "이번달", "今月", "本月"), 30, 0),
    ]
    for words, back_start, back_end in table:
        if any(w in ql for w in words):
            start = _day(now) - timedelta(days=back_start)
            end = now if back_end == 0 else _day(now) - timedelta(days=back_end) + timedelta(days=1)
            return start, end
    m = _YM.search(q)
    if m:
        y, mo = int(m[1]), int(m[2])
        if 1 <= mo <= 12:
            start = datetime(y, mo, 1, tzinfo=timezone.utc)
            end = datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc)
            return start, end
    return None


def find_date(text: str) -> Optional[datetime]:
    m = _ISO.search(text or "")
    if not m:
        return None
    try:
        return datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
    except ValueError:
        return None


_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def split_sentences(text: str, min_len: int = 8) -> list[str]:
    parts = [p.strip() for p in _SENT.split(text or "") if p and p.strip()]
    return [p for p in parts if len(p) >= min_len] or ([text.strip()] if text and text.strip() else [])


def approx_tokens(text: str) -> int:
    """Rough token count: 4 chars/token for Latin, 1 char/token for CJK."""
    cjk = len(_CJK.findall(text or ""))
    return cjk + max(0, len(text or "") - cjk) // 4 + 1
