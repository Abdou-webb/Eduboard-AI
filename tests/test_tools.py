import pytest

from eduboard.mathcheck import MathError, check_math
from eduboard.memory import LessonStore
from eduboard.preprocess import scan
from eduboard.synthetic import make_fake_board
from eduboard.tools import TOOLS, Toolbox


# --- check_math ----------------------------------------------------------

def test_derivative_of_product():
    out = check_math("derivative", "x^2 sin(x)", "2x sin(x) + x^2 cos(x)")
    assert "CORRECT" in out and "INCORRECT" not in out


def test_detects_wrong_derivative():
    out = check_math("derivative", "x^2 sin(x)", "2x cos(x)")
    assert "INCORRECT" in out


def test_equals_with_equation_syntax():
    assert check_math("equals", "(a+b)^2 = a^2 + 2a b + b^2").startswith("VRAI")
    assert check_math("equals", "(a+b)^2 = a^2 + b^2").startswith("FAUX")


def test_solve():
    out = check_math("solve", "x^2 - 5x + 6 = 0")
    assert "2" in out and "3" in out


def test_integral_accepts_any_constant():
    assert "CORRECTE" in check_math("integral", "2x", "x^2 + 7")


def test_bad_expression():
    with pytest.raises(MathError):
        check_math("simplify", "x +* (")
    with pytest.raises(MathError):
        check_math("factorise", "x")


# --- memoire -------------------------------------------------------------

def test_store_search(tmp_path):
    store = LessonStore(tmp_path / "l.json")
    store.add({"title": "Derivees usuelles", "course": "Analyse", "key_concepts": ["derivee", "tangente"], "explanation": ""})
    store.add({"title": "Integrales", "course": "Analyse", "key_concepts": ["primitive"], "explanation": "aire sous la courbe"})
    store.add({"title": "Modele OSI", "course": "Reseaux", "key_concepts": ["couches"], "explanation": ""})

    hits = store.search("dérivée d'un produit", course="Analyse")
    assert hits and hits[0][1]["title"] == "Derivees usuelles"
    assert store.search("couches", course="Analyse") == []

    # persistance
    assert len(LessonStore(tmp_path / "l.json").all()) == 3


# --- toolbox -------------------------------------------------------------

@pytest.fixture
def toolbox(tmp_path):
    return Toolbox(scan(make_fake_board()), LessonStore(tmp_path / "l.json"), course="Analyse")


def test_tool_schemas_are_strict():
    for t in TOOLS:
        schema = t["input_schema"]
        assert t["strict"] and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_zoom_returns_image(toolbox):
    content, err = toolbox.run("zoom_region", {"x": 0, "y": 0, "w": .5, "h": .3, "source": "clean", "reason": "titre"})
    assert not err
    assert content[0]["type"] == "image"


def test_zoom_limit(toolbox):
    args = {"x": 0, "y": 0, "w": .5, "h": .5, "source": "original", "reason": ""}
    for _ in range(8):
        toolbox.run("zoom_region", args)
    _, err = toolbox.run("zoom_region", args)
    assert err


def test_unknown_tool(toolbox):
    _, err = toolbox.run("rm_rf", {})
    assert err


def _lesson(**over):
    base = dict(title="Derivee d'un produit", subject="Analyse", transcription="$(uv)'=u'v+uv'$",
                explanation="...", key_concepts=["derivee"], warnings=[],
                quiz=[{"question": "?", "choices": ["a", "b"], "answer_index": 1, "explanation": "b"}])
    base.update(over)
    return base


def test_save_lesson(toolbox):
    out, err = toolbox.run("save_lesson", _lesson())
    assert not err and toolbox.lesson["course"] == "Analyse"
    # deuxieme appel refuse
    _, err = toolbox.run("save_lesson", _lesson())
    assert err


def test_save_lesson_rejects_bad_quiz(toolbox):
    bad = _lesson(quiz=[{"question": "?", "choices": ["a"], "answer_index": 3, "explanation": ""}])
    _, err = toolbox.run("save_lesson", bad)
    assert err and toolbox.lesson is None
