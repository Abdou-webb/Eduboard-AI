"""Memoire long terme de l'agent : les lecons deja digitalisees.

Volontairement simple (un fichier JSON + recherche par mots-cles) :
pour quelques centaines de lecons pas besoin d'une base vectorielle.
"""
import json
import re
import unicodedata
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

STOPWORDS = {
    "les", "des", "une", "est", "que", "qui", "dans", "pour", "par", "sur", "avec",
    "pas", "plus", "son", "ses", "aux", "elle", "ils", "cette", "mais", "donc",
    "the", "and", "for", "with", "this", "that",
}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return [t for t in re.findall(r"[a-z0-9]{3,}", text) if t not in STOPWORDS]


class LessonStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lessons: list[dict] = []
        if self.path.exists():
            self._lessons = json.loads(self.path.read_text(encoding="utf-8"))

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._lessons, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def all(self, course: str | None = None) -> list[dict]:
        if not course:
            return list(self._lessons)
        return [l for l in self._lessons if l.get("course", "").lower() == course.lower()]

    def get(self, lesson_id: str) -> dict | None:
        return next((l for l in self._lessons if l["id"] == lesson_id), None)

    def add(self, lesson: dict) -> dict:
        lesson = dict(lesson)
        lesson.setdefault("id", uuid.uuid4().hex[:8])
        lesson.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        self._lessons.append(lesson)
        self._flush()
        return lesson

    def delete(self, lesson_id: str) -> bool:
        before = len(self._lessons)
        self._lessons = [l for l in self._lessons if l["id"] != lesson_id]
        if len(self._lessons) != before:
            self._flush()
            return True
        return False

    def search(self, query: str, course: str | None = None, k: int = 3) -> list[tuple[float, dict]]:
        q = set(tokenize(query))
        if not q:
            return []
        scored = []
        for lesson in self.all(course):
            # le titre et les concepts cles comptent plus que le corps
            counts = Counter()
            for tok in tokenize(lesson.get("title", "")):
                counts[tok] += 3
            for concept in lesson.get("key_concepts", []):
                for tok in tokenize(concept):
                    counts[tok] += 2
            for tok in tokenize(lesson.get("explanation", "")):
                counts[tok] += 1
            score = sum(counts[t] for t in q)
            if score > 0:
                scored.append((score, lesson))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
