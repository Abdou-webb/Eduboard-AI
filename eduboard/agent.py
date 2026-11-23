"""Boucle de l'agent (perception -> raisonnement -> action -> observation).

On ecrit la boucle a la main plutot que d'utiliser un framework : ca permet
de voir exactement ce qui se passe a chaque tour, d'ajouter nos garde-fous
(nombre max d'etapes, relance si le modele oublie de sauvegarder) et
d'afficher la trace dans l'interface.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable

from . import prompts
from .config import Settings
from .memory import LessonStore
from .preprocess import BoardScan, to_jpeg_b64
from .tools import TOOLS, Toolbox

log = logging.getLogger("eduboard.agent")

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AgentError(RuntimeError):
    pass


@dataclass
class Step:
    kind: str     # "thinking" | "text" | "tool_call" | "tool_result" | "info"
    title: str
    detail: str = ""
    is_error: bool = False


@dataclass
class AgentResult:
    lesson: dict | None
    steps: list[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0


def _short(value, n=300) -> str:
    s = str(value)
    return s if len(s) <= n else s[:n] + "..."


class EduBoardAgent:
    def __init__(
        self,
        client=None,
        settings: Settings | None = None,
        store: LessonStore | None = None,
        on_step: Callable[[Step], None] | None = None,
    ):
        self.settings = settings or Settings()
        if client is None:
            client = self._make_client()
        self.client = client
        self.store = store or LessonStore(self.settings.lessons_path)
        self.on_step = on_step

    # ------------------------------------------------------------------
    def _emit(self, result: AgentResult, step: Step):
        result.steps.append(step)
        log.info("[%s] %s", step.kind, step.title)
        if self.on_step:
            self.on_step(step)

    def _make_client(self):
        # imports tardifs : les tests n'en ont pas besoin
        if self.settings.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic()
        import os
        from openai import OpenAI
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise AgentError("GROQ_API_KEY manquante dans .env")
        # max_retries : le palier gratuit de Groq renvoie souvent 429 (limite par minute)
        return OpenAI(api_key=key, base_url=self.settings.groq_base_url, max_retries=5)

    def _call_model(self, messages):
        if self.settings.provider != "anthropic":
            return self._call_openai(messages)
        kwargs = dict(
            model=self.settings.model,
            max_tokens=16000,
            system=prompts.SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.settings.effort},
            # l'image et le prompt systeme sont renvoyes a chaque tour -> cache
            cache_control={"type": "ephemeral"},
        )
        if self.settings.fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return self.client.beta.messages.create(**kwargs)

    def _call_openai(self, messages):
        from openai import BadRequestError

        from . import openai_compat as oc
        msgs = oc.convert_messages(prompts.SYSTEM_PROMPT, messages, self.settings.max_images)
        names = ", ".join(t["name"] for t in TOOLS)
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.settings.groq_model,
                    max_tokens=8192,
                    temperature=0.2,
                    tools=oc.convert_tools(TOOLS),
                    messages=msgs,
                )
                return oc.convert_response(resp)
            except BadRequestError as e:
                # Groq valide les appels d'outils cote serveur : nom invente ou JSON
                # invalide -> 400 "tool_use_failed". On explique l'erreur et on relance.
                body = e.body if isinstance(e.body, dict) else {}
                err = body.get("error", body) if isinstance(body.get("error", body), dict) else {}
                if err.get("code") != "tool_use_failed" or attempt == 2:
                    raise
                log.warning("appel d'outil invalide, relance : %s", err.get("message"))
                msgs = msgs + [{
                    "role": "user",
                    "content": f"Ton appel d'outil etait invalide ({err.get('message', '')}). "
                               f"Outils disponibles exactement : {names}. Reessaie avec un nom et des arguments valides.",
                }]

    # ------------------------------------------------------------------
    def run(self, scan: BoardScan, course: str = "", hint: str = "") -> AgentResult:
        toolbox = Toolbox(scan, self.store, course)
        result = AgentResult(lesson=None)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": to_jpeg_b64(scan.clean)},
                },
                {"type": "text", "text": prompts.first_message(course, hint, scan.board_found, scan.inverted)},
            ],
        }]

        nudged = False
        for turn in range(1, self.settings.max_steps + 1):
            result.turns = turn
            resp = self._call_model(messages)
            if resp.usage:
                result.input_tokens += (resp.usage.input_tokens or 0)
                result.output_tokens += (resp.usage.output_tokens or 0)

            if resp.stop_reason == "refusal":
                raise AgentError("Le modele a refuse la requete.")

            # on renvoie le contenu tel quel (blocs de reflexion compris)
            messages.append({"role": "assistant", "content": resp.content})

            tool_calls = []
            for block in resp.content:
                if block.type == "thinking" and getattr(block, "thinking", ""):
                    self._emit(result, Step("thinking", "Reflexion", block.thinking))
                elif block.type == "text" and block.text.strip():
                    self._emit(result, Step("text", "Message", block.text))
                elif block.type == "tool_use":
                    tool_calls.append(block)

            if resp.stop_reason == "max_tokens":
                self._emit(result, Step("info", "Reponse coupee (max_tokens)", is_error=True))
                break

            if not tool_calls:
                if toolbox.lesson is not None or nudged:
                    break
                # le modele s'arrete sans sauvegarder : on le relance une fois
                nudged = True
                self._emit(result, Step("info", "Relance : save_lesson manquant"))
                messages.append({"role": "user", "content": prompts.NUDGE_SAVE})
                continue

            tool_results = []
            for call in tool_calls:
                self._emit(result, Step("tool_call", call.name, _short(call.input)))
                content, is_error = toolbox.run(call.name, call.input)
                shown = "[image]" if isinstance(content, list) else _short(content)
                self._emit(result, Step("tool_result", call.name, shown, is_error))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": content,
                    "is_error": is_error,
                })
            # tous les resultats dans UN seul message user
            messages.append({"role": "user", "content": tool_results})

            if toolbox.lesson is not None:
                # inutile de refaire un tour juste pour "c'est enregistre"
                break
        else:
            self._emit(result, Step("info", f"Arret : {self.settings.max_steps} etapes atteintes", is_error=True))

        result.lesson = toolbox.lesson
        return result
