import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv est optionnel
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "oui")


def _default_provider() -> str:
    # explicite > cle Groq presente > Anthropic
    p = os.getenv("EDUBOARD_PROVIDER")
    if p:
        return p.strip().lower()
    return "groq" if os.getenv("GROQ_API_KEY") else "anthropic"


@dataclass
class Settings:
    # "anthropic" ou "groq" (API compatible OpenAI)
    provider: str = field(default_factory=_default_provider)
    model: str = field(default_factory=lambda: os.getenv("EDUBOARD_MODEL", "claude-opus-5-5"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    # certains modeles limitent le nombre d'images par requete : on garde les plus recentes
    max_images: int = field(default_factory=lambda: int(os.getenv("EDUBOARD_MAX_IMAGES", "5")))
    # low / medium / high : plus c'est haut, plus l'agent reflechit (et coute)
    effort: str = field(default_factory=lambda: os.getenv("EDUBOARD_EFFORT", "medium"))
    max_steps: int = field(default_factory=lambda: int(os.getenv("EDUBOARD_MAX_STEPS", "12")))
    # repli automatique sur un autre modele si une requete est refusee
    fallback: bool = field(default_factory=lambda: _env_bool("EDUBOARD_FALLBACK", True))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("EDUBOARD_DATA", ROOT / "data")))

    @property
    def lessons_path(self) -> Path:
        return self.data_dir / "lessons.json"
