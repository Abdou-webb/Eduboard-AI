"""Les outils que l'agent peut appeler + leur implementation."""
from .mathcheck import OPERATIONS, MathError, check_math
from .memory import LessonStore
from .preprocess import BoardScan, crop_normalized, to_jpeg_b64

TOOLS = [
    {
        "name": "zoom_region",
        "description": (
            "Renvoie une vue agrandie d'une zone du tableau. Coordonnees relatives "
            "entre 0 et 1 (x, y = coin haut-gauche ; w, h = largeur, hauteur). "
            "source='clean' pour l'image nettoyee, 'original' pour la photo brute "
            "redressee (meilleure pour les couleurs et les traits tres fins)."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "w": {"type": "number"},
                "h": {"type": "number"},
                "source": {"type": "string", "enum": ["clean", "original"]},
                "reason": {"type": "string", "description": "Pourquoi tu zoomes ici (une phrase)."},
            },
            "required": ["x", "y", "w", "h", "source", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_math",
        "description": (
            "Verifie un calcul avec un moteur de calcul formel (sympy). Syntaxe "
            "sympy/python, PAS du LaTeX : x**2 ou x^2, sin(x), sqrt(x), exp(x), pi. "
            "Operations : simplify, equals (expression = expected, ou 'a = b' dans "
            "expression), solve (equation en 'variable'), derivative et integral "
            "(par rapport a 'variable', 'expected' optionnel pour comparer), evaluate. "
            "Mettre une chaine vide pour les champs inutiles."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": list(OPERATIONS)},
                "expression": {"type": "string"},
                "expected": {"type": "string"},
                "variable": {"type": "string"},
            },
            "required": ["operation", "expression", "expected", "variable"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_course_notes",
        "description": (
            "Cherche dans les lecons deja digitalisees du meme cours (memoire de "
            "l'agent). Renvoie les lecons les plus proches avec leurs concepts cles."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "save_lesson",
        "description": "Enregistre la lecon finale. A appeler une seule fois, a la fin.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "subject": {"type": "string", "description": "Matiere, ex: Analyse, Physique, Reseaux..."},
                "transcription": {"type": "string", "description": "Contenu du tableau en Markdown + LaTeX."},
                "explanation": {"type": "string", "description": "Explication pedagogique en Markdown."},
                "key_concepts": {"type": "array", "items": {"type": "string"}},
                "warnings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Erreurs reperees au tableau, passages illisibles, doutes.",
                },
                "quiz": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "choices": {"type": "array", "items": {"type": "string"}},
                            "answer_index": {"type": "integer"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["question", "choices", "answer_index", "explanation"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "title", "subject", "transcription", "explanation",
                "key_concepts", "warnings", "quiz",
            ],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    pass


class Toolbox:
    """Execute les appels d'outils pour UNE analyse de tableau."""

    def __init__(self, scan: BoardScan, store: LessonStore, course: str = ""):
        self.scan = scan
        self.store = store
        self.course = course
        self.lesson: dict | None = None
        self.zoom_count = 0

    def run(self, name: str, args: dict):
        """Renvoie (contenu, is_error). Le contenu est une str ou une liste de blocs."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Outil inconnu : {name}", True
        try:
            return handler(**args), False
        except (ToolError, MathError, ValueError, TypeError) as e:
            return f"Erreur : {e}", True

    def _tool_zoom_region(self, x, y, w, h, source="clean", reason=""):
        if self.zoom_count >= 8:
            # garde-fou : sinon l'agent peut zoomer en boucle (et chaque image coute)
            raise ToolError("limite de zooms atteinte, travaille avec ce que tu as")
        img = self.scan.clean if source == "clean" else self.scan.warped
        crop = crop_normalized(img, x, y, w, h)
        self.zoom_count += 1
        return [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": to_jpeg_b64(crop, 90)},
            },
            {"type": "text", "text": f"Zone x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f} ({source})"},
        ]

    def _tool_check_math(self, operation, expression, expected="", variable="x"):
        return check_math(operation, expression, expected, variable)

    def _tool_search_course_notes(self, query):
        hits = self.store.search(query, course=self.course or None)
        if not hits:
            return "Aucune lecon precedente ne correspond."
        out = []
        for score, l in hits:
            concepts = ", ".join(l.get("key_concepts", [])[:6])
            out.append(f"- [{l.get('created_at', '?')[:10]}] {l['title']} (score {score}) -- concepts : {concepts}")
        return "\n".join(out)

    def _tool_save_lesson(self, **lesson):
        if self.lesson is not None:
            raise ToolError("lecon deja enregistree")
        for i, q in enumerate(lesson.get("quiz", [])):
            if not 0 <= q["answer_index"] < len(q["choices"]):
                raise ToolError(f"question {i + 1} : answer_index hors limites")
        if not lesson["transcription"].strip():
            raise ToolError("transcription vide")
        lesson["course"] = self.course
        self.lesson = self.store.add(lesson)
        return f"Lecon enregistree (id {self.lesson['id']})."
