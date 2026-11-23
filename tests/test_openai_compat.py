"""Teste l'adaptateur OpenAI/Groq et la boucle de l'agent avec un faux client OpenAI."""
import json
from types import SimpleNamespace as NS

from eduboard import openai_compat as oc
from eduboard.agent import EduBoardAgent
from eduboard.config import Settings
from eduboard.memory import LessonStore
from eduboard.preprocess import scan
from eduboard.synthetic import make_fake_board
from eduboard.tools import TOOLS

from test_agent import LESSON


def oai_response(tool_calls=None, content=None, finish="tool_calls"):
    calls = [NS(id=i, function=NS(name=n, arguments=json.dumps(a))) for i, n, a in (tool_calls or [])]
    msg = NS(content=content, tool_calls=calls or None, reasoning=None)
    return NS(choices=[NS(message=msg, finish_reason=finish)], usage=NS(prompt_tokens=50, completion_tokens=10))


class FakeOpenAI:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def test_convert_tools():
    out = oc.convert_tools(TOOLS)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["parameters"] == TOOLS[0]["input_schema"]


def test_full_run_groq(tmp_path):
    s = Settings(data_dir=tmp_path, provider="groq")
    client = FakeOpenAI([
        oai_response([("c1", "zoom_region", {"x": .1, "y": .3, "w": .8, "h": .2, "source": "clean", "reason": "formule"})]),
        oai_response([("c2", "save_lesson", LESSON)]),
    ])
    agent = EduBoardAgent(client=client, settings=s, store=LessonStore(s.lessons_path))
    res = agent.run(scan(make_fake_board()), course="Analyse")

    assert res.lesson is not None and res.input_tokens == 100
    msgs = client.calls[1]["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assert msgs[2]["tool_calls"][0]["id"] == "c1"
    assert msgs[3]["tool_call_id"] == "c1"
    # l'image du zoom est renvoyee dans un message user
    assert any(p["type"] == "image_url" for p in msgs[4]["content"])


def test_image_limit_keeps_first_and_latest():
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAA"}}
    messages = [{"role": "user", "content": [img, {"type": "text", "text": "go"}]}]
    for k in range(4):
        messages.append({"role": "assistant", "content": [NS(type="tool_use", id=f"z{k}", name="zoom_region", input={})]})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"z{k}", "content": [img], "is_error": False}]})
    out = oc.convert_messages("sys", messages, max_images=3)
    images = [p for m in out if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url"]
    assert len(images) == 3
    assert out[1]["content"][0]["type"] == "image_url"  # tableau entier conserve


def test_invalid_json_arguments_become_error():
    msg = NS(content=None, tool_calls=[NS(id="x", function=NS(name="check_math", arguments="{oops"))], reasoning=None)
    r = oc.convert_response(NS(choices=[NS(message=msg, finish_reason="tool_calls")], usage=None))
    assert r.stop_reason == "tool_use" and "_invalid_json" in r.content[0].input
