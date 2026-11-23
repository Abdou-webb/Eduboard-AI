"""Interface web : streamlit run app.py"""
import os

import streamlit as st

from eduboard.agent import AgentError, EduBoardAgent, Step
from eduboard.config import Settings
from eduboard.export import lesson_to_markdown
from eduboard.memory import LessonStore
from eduboard.preprocess import scan, to_rgb

st.set_page_config(page_title="EduBoard AI", page_icon=":pencil:", layout="wide")

settings = Settings()


@st.cache_resource
def get_store():
    return LessonStore(settings.lessons_path)


store = get_store()

ICONS = {"thinking": "🧠", "text": "💬", "tool_call": "🔧", "tool_result": "📥", "info": "ℹ️"}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("EduBoard AI")
    st.caption("Digitalisation et explication de tableau blanc")

    course = st.text_input("Cours", placeholder="ex : Analyse 1, Reseaux, Thermo...")
    hint = st.text_area("Remarque (optionnel)", placeholder="ex : le prof a parle du theoreme de Rolle", height=80)
    settings.effort = st.select_slider("Effort de raisonnement", ["low", "medium", "high"], value=settings.effort)

    if settings.provider == "anthropic":
        st.caption(f"Modele : {settings.model} (Anthropic)")
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            st.warning("Pas de cle API detectee (ANTHROPIC_API_KEY). Seul le pretraitement marchera.")
    else:
        st.caption(f"Modele : {settings.groq_model} ({settings.provider})")
        if not os.getenv("GROQ_API_KEY"):
            st.warning("Pas de cle API detectee (GROQ_API_KEY). Seul le pretraitement marchera.")

    st.divider()
    st.subheader("Historique")
    lessons = store.all(course or None)
    if not lessons:
        st.caption("Aucune lecon pour l'instant.")
    for l in reversed(lessons[-15:]):
        if st.button(f"{l['title'][:40]}", key=f"hist_{l['id']}", width="stretch"):
            st.session_state["lesson"] = l
            st.session_state["steps"] = []

# ---------------------------------------------------------------- main
uploaded = st.file_uploader("Photo du tableau", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    data = uploaded.getvalue()
    if st.session_state.get("upload_name") != uploaded.name:
        st.session_state["upload_name"] = uploaded.name
        st.session_state["scan"] = scan(data)
        st.session_state.pop("lesson", None)

    sc = st.session_state["scan"]
    c1, c2 = st.columns(2)
    c1.image(to_rgb(sc.original), caption="Photo d'origine", width="stretch")
    caption = "Redresse + nettoye" if sc.board_found else "Nettoye (bords du tableau non detectes)"
    c2.image(to_rgb(sc.clean), caption=caption, width="stretch")

    if st.button("Analyser le tableau", type="primary"):
        trace_box = st.status("L'agent travaille...", expanded=True)

        def show_step(step: Step):
            trace_box.write(f"{ICONS.get(step.kind, '')} **{step.title}** {('— ' + step.detail[:200]) if step.detail else ''}")

        agent = EduBoardAgent(settings=settings, store=store, on_step=show_step)
        try:
            res = agent.run(sc, course=course, hint=hint)
        except AgentError as e:
            trace_box.update(label=str(e), state="error")
            st.stop()
        except Exception as e:  # erreur reseau, cle invalide...
            trace_box.update(label=f"Erreur : {e}", state="error")
            st.stop()

        trace_box.update(
            label=f"Termine en {res.turns} tours — {res.input_tokens} tokens in / {res.output_tokens} out",
            state="complete" if res.lesson else "error",
            expanded=False,
        )
        st.session_state["steps"] = res.steps
        if res.lesson:
            st.session_state["lesson"] = res.lesson
        else:
            st.error("L'agent n'a pas produit de lecon. Voir la trace ci-dessus.")

# ---------------------------------------------------------------- resultat
lesson = st.session_state.get("lesson")
if lesson:
    st.divider()
    st.title(lesson["title"])
    st.caption(" · ".join(x for x in (lesson.get("course"), lesson.get("subject")) if x))

    if lesson.get("warnings"):
        with st.expander(f"⚠️ Points a verifier ({len(lesson['warnings'])})", expanded=True):
            for w in lesson["warnings"]:
                st.markdown(f"- {w}")

    t1, t2, t3, t4 = st.tabs(["Tableau", "Explication", "Quiz", "Trace de l'agent"])
    with t1:
        st.markdown(lesson["transcription"])
    with t2:
        st.markdown(lesson["explanation"])
        if lesson.get("key_concepts"):
            st.markdown("**Concepts cles :** " + ", ".join(lesson["key_concepts"]))
    with t3:
        for i, q in enumerate(lesson.get("quiz", [])):
            st.markdown(f"**{i + 1}. {q['question']}**")
            choice = st.radio("Reponse", q["choices"], index=None, key=f"q{lesson['id']}_{i}", label_visibility="collapsed")
            if choice is not None:
                if q["choices"].index(choice) == q["answer_index"]:
                    st.success("Correct ! " + q["explanation"])
                else:
                    st.error("Non. " + q["explanation"])
    with t4:
        steps = st.session_state.get("steps") or []
        if not steps:
            st.caption("Pas de trace (lecon chargee depuis l'historique).")
        for s in steps:
            with st.expander(f"{ICONS.get(s.kind, '')} {s.title}", expanded=False):
                st.text(s.detail or "-")

    st.download_button(
        "Telecharger en Markdown",
        lesson_to_markdown(lesson),
        file_name=f"{lesson['title'][:40].replace(' ', '_')}.md",
        mime="text/markdown",
    )
