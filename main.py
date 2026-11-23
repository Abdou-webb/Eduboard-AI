"""Ligne de commande.

    python main.py clean photo.jpg -o propre.jpg        (pas besoin de cle API)
    python main.py analyse photo.jpg --course "Analyse 1" -o lecon.md
    python main.py history --course "Analyse 1"
"""
import argparse
import logging
import sys

import cv2

from eduboard.config import Settings
from eduboard.memory import LessonStore
from eduboard.preprocess import scan


def cmd_clean(args):
    sc = scan(args.image)
    cv2.imwrite(args.output, sc.clean)
    print(f"tableau detecte : {'oui' if sc.board_found else 'non'} | inverse : {sc.inverted}")
    print("->", args.output)


def cmd_analyse(args):
    from eduboard.agent import EduBoardAgent
    from eduboard.export import lesson_to_markdown

    settings = Settings()
    if args.effort:
        settings.effort = args.effort

    def printer(step):
        tag = {"tool_call": ">>", "tool_result": "<<", "thinking": "..", "text": "  ", "info": "!!"}[step.kind]
        line = f"{tag} {step.title}"
        if step.detail and step.kind != "thinking":
            line += f" : {step.detail[:120]}"
        print(line.replace("\n", " "))

    agent = EduBoardAgent(settings=settings, on_step=printer if not args.quiet else None)
    res = agent.run(scan(args.image), course=args.course, hint=args.hint)

    print(f"\n{res.turns} tours, {res.input_tokens} tokens in / {res.output_tokens} out")
    if not res.lesson:
        print("Echec : pas de lecon produite.", file=sys.stderr)
        sys.exit(1)

    md = lesson_to_markdown(res.lesson)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        print("->", args.output)
    else:
        print("\n" + md)


def cmd_history(args):
    store = LessonStore(Settings().lessons_path)
    lessons = store.all(args.course)
    if not lessons:
        print("rien.")
    for l in lessons:
        print(f"{l['id']}  {l.get('created_at', '')[:10]}  [{l.get('course') or '-'}]  {l['title']}")


def main():
    p = argparse.ArgumentParser(prog="eduboard")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clean", help="redresse et nettoie une photo")
    c.add_argument("image")
    c.add_argument("-o", "--output", default="clean.jpg")
    c.set_defaults(func=cmd_clean)

    a = sub.add_parser("analyse", help="lance l'agent sur une photo")
    a.add_argument("image")
    a.add_argument("--course", default="")
    a.add_argument("--hint", default="")
    a.add_argument("--effort", choices=["low", "medium", "high"])
    a.add_argument("-o", "--output")
    a.add_argument("-q", "--quiet", action="store_true")
    a.set_defaults(func=cmd_analyse)

    h = sub.add_parser("history", help="liste les lecons enregistrees")
    h.add_argument("--course")
    h.set_defaults(func=cmd_history)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
