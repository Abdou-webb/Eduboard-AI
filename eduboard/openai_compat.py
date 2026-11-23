"""Adaptateur pour les API compatibles OpenAI (Groq, OpenRouter, Ollama...).

La boucle de l'agent travaille avec des messages au format Anthropic
(blocs text / image / tool_use / tool_result). Ici on :
- convertit ces messages et les outils au format OpenAI avant l'appel ;
- reconvertit la reponse en objet "facon Anthropic" (content, stop_reason, usage),
  pour que agent.py n'ait pas a changer.
"""
import json
from types import SimpleNamespace as NS

STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn", "content_filter": "refusal"}


def convert_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in tools
    ]


def _image_part(block: dict) -> dict:
    src = block["source"]
    return {"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}}


def _get(block, key, default=None):
    # les blocs assistant sont des objets, les blocs user des dicts
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def convert_messages(system: str, messages: list[dict], max_images: int = 5) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]

    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue

        if msg["role"] == "assistant":
            texts, calls = [], []
            for b in content:
                kind = _get(b, "type")
                if kind == "text":
                    texts.append(_get(b, "text", ""))
                elif kind == "tool_use":
                    calls.append({
                        "id": _get(b, "id"),
                        "type": "function",
                        "function": {"name": _get(b, "name"), "arguments": json.dumps(_get(b, "input"), ensure_ascii=False)},
                    })
            m = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                m["tool_calls"] = calls
            out.append(m)
            continue

        # message user : texte/images, ou resultats d'outils
        parts, extra_images = [], []
        for b in content:
            kind = b["type"]
            if kind == "text":
                parts.append({"type": "text", "text": b["text"]})
            elif kind == "image":
                parts.append(_image_part(b))
            elif kind == "tool_result":
                res = b["content"]
                if isinstance(res, list):
                    # un message "tool" ne peut contenir que du texte : les images
                    # (zoom) sont renvoyees juste apres dans un message user
                    txt = " ".join(x["text"] for x in res if x["type"] == "text")
                    extra_images += [_image_part(x) for x in res if x["type"] == "image"]
                    res = (txt + " (image jointe dans le message suivant)").strip()
                if b.get("is_error"):
                    res = f"[ERREUR] {res}"
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": res})
        if parts:
            out.append({"role": "user", "content": parts})
        if extra_images:
            out.append({"role": "user", "content": [{"type": "text", "text": "Image(s) du zoom :"}, *extra_images]})

    _limit_images(out, max_images)
    return out


def _limit_images(messages: list[dict], max_images: int):
    """Garde la premiere image (le tableau entier) et les plus recentes."""
    slots = [
        (i, j)
        for i, m in enumerate(messages)
        if isinstance(m.get("content"), list)
        for j, p in enumerate(m["content"])
        if p.get("type") == "image_url"
    ]
    if len(slots) <= max_images:
        return
    keep = {slots[0]} | set(slots[-(max_images - 1):]) if max_images > 1 else {slots[0]}
    for slot in slots:
        if slot not in keep:
            i, j = slot
            messages[i]["content"][j] = {"type": "text", "text": "[ancienne image retiree pour respecter la limite]"}


def convert_response(resp) -> NS:
    choice = resp.choices[0]
    msg = choice.message
    blocks = []
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if reasoning:
        blocks.append(NS(type="thinking", thinking=reasoning))
    if msg.content:
        blocks.append(NS(type="text", text=msg.content))
    for call in msg.tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            # arguments invalides : l'outil renverra une erreur au modele
            args = {"_invalid_json": call.function.arguments}
        blocks.append(NS(type="tool_use", id=call.id, name=call.function.name, input=args))

    stop = STOP_REASONS.get(choice.finish_reason, "end_turn")
    if msg.tool_calls and stop == "end_turn":
        stop = "tool_use"
    usage = resp.usage
    return NS(
        content=blocks,
        stop_reason=stop,
        usage=NS(input_tokens=getattr(usage, "prompt_tokens", 0), output_tokens=getattr(usage, "completion_tokens", 0)) if usage else None,
    )
