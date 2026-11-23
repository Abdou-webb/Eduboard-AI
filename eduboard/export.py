def lesson_to_markdown(lesson: dict, with_answers: bool = True) -> str:
    out = [f"# {lesson['title']}", ""]
    meta = [m for m in (lesson.get("course"), lesson.get("subject"), (lesson.get("created_at") or "")[:10]) if m]
    if meta:
        out += [" | ".join(meta), ""]

    out += ["## Contenu du tableau", "", lesson["transcription"].strip(), ""]
    out += ["## Explication", "", lesson["explanation"].strip(), ""]

    if lesson.get("key_concepts"):
        out += ["## Concepts cles", ""]
        out += [f"- {c}" for c in lesson["key_concepts"]]
        out.append("")

    if lesson.get("warnings"):
        out += ["## A verifier", ""]
        out += [f"- {w}" for w in lesson["warnings"]]
        out.append("")

    quiz = lesson.get("quiz") or []
    if quiz:
        out += ["## Quiz", ""]
        letters = "ABCDEFGH"
        for i, q in enumerate(quiz, 1):
            out.append(f"**{i}. {q['question']}**")
            out.append("")
            for j, choice in enumerate(q["choices"]):
                out.append(f"- {letters[j]}) {choice}")
            out.append("")
        if with_answers:
            out += ["### Corrige", ""]
            for i, q in enumerate(quiz, 1):
                out.append(f"{i}. **{letters[q['answer_index']]}** - {q['explanation']}")
            out.append("")

    return "\n".join(out)
