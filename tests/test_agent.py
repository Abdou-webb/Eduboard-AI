"""Teste la boucle de l'agent sans appeler l'API (faux client scripte)."""
from types import SimpleNamespace as NS

from eduboard.agent import EduBoardAgent
from eduboard.config import Settings
from eduboard.memory import LessonStore
from eduboard.preprocess import scan
from eduboard.synthetic import make_fake_board

LESSON = dict(
    title="Derivee d'un produit", subject="Analyse",
    transcription="$(uv)' = u'v + uv'$", explanation="On derive chaque facteur.",
    key_concepts=["derivee", "produit"], warnings=[],
    quiz=[{"question": "(x sin x)' ?", "choices": ["sin x + x cos x", "cos x"], "answer_index": 0, "explanation": "regle du produit"}],
)


def tool_use(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


def response(blocks, stop="tool_use"):
    return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=100, output_tokens=20))


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        # copie de la liste : l'agent la modifie apres l'appel
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


def make_agent(tmp_path, script, **settings):
    s = Settings(data_dir=tmp_path, **{"provider": "anthropic", **settings})
    client = FakeClient(script)
    return EduBoardAgent(client=client, settings=s, store=LessonStore(s.lessons_path)), client


def test_full_run(tmp_path):
    agent, client = make_agent(tmp_path, [
        response([text("Je zoome sur la formule."),
                  tool_use("t1", "zoom_region", {"x": .1, "y": .3, "w": .8, "h": .2, "source": "clean", "reason": "formule"})]),
        response([tool_use("t2", "check_math", {"operation": "derivative", "expression": "x^2 sin(x)",
                                                 "expected": "2x sin(x) + x^2 cos(x)", "variable": "x"})]),
        response([tool_use("t3", "save_lesson", LESSON)]),
    ])
    res = agent.run(scan(make_fake_board()), course="Analyse")

    assert res.lesson is not None and res.lesson["title"] == LESSON["title"]
    assert res.turns == 3
    assert res.input_tokens == 300
    # le resultat de check_math est bien renvoye au modele
    last_msgs = client.calls[2]["messages"]
    assert "CORRECT" in last_msgs[-1]["content"][0]["content"]
    # memoire persistee
    assert len(LessonStore(tmp_path / "lessons.json").all()) == 1


def test_nudge_when_model_forgets_to_save(tmp_path):
    agent, client = make_agent(tmp_path, [
        response([text("Voici l'explication...")], stop="end_turn"),
        response([tool_use("t1", "save_lesson", LESSON)]),
    ])
    res = agent.run(scan(make_fake_board()))
    assert res.lesson is not None
    assert client.calls[1]["messages"][-1]["content"].startswith("Tu n'as pas")


def test_tool_error_is_sent_back(tmp_path):
    agent, client = make_agent(tmp_path, [
        response([tool_use("t1", "check_math", {"operation": "simplify", "expression": "((", "expected": "", "variable": "x"})]),
        response([tool_use("t2", "save_lesson", LESSON)]),
    ])
    agent.run(scan(make_fake_board()))
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_max_steps(tmp_path):
    loop = [response([tool_use(f"t{i}", "search_course_notes", {"query": "x"})]) for i in range(3)]
    agent, _ = make_agent(tmp_path, loop, max_steps=3)
    res = agent.run(scan(make_fake_board()))
    assert res.lesson is None
    assert res.steps[-1].is_error


def test_request_params(tmp_path):
    agent, client = make_agent(tmp_path, [response([tool_use("t1", "save_lesson", LESSON)])], fallback=False)
    agent.run(scan(make_fake_board()))
    call = client.calls[0]
    assert call["thinking"]["type"] == "adaptive"
    assert "fallbacks" not in call
    assert call["messages"][0]["content"][0]["type"] == "image"
