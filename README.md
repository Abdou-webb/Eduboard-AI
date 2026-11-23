# EduBoard AI

Agent de digitalisation et d'explication de tableau blanc.

On prend en photo le tableau à la fin du cours, et l'agent :
1. redresse et nettoie la photo (OpenCV),
2. retranscrit le contenu (texte, formules en LaTeX, description des schémas),
3. vérifie les calculs avec un moteur de calcul formel (sympy),
4. fait le lien avec les séances précédentes du même cours,
5. rédige une explication et un petit quiz.

Projet réalisé dans le cadre du module **Agentic AI**.

## Pourquoi un agent et pas un simple appel au LLM ?

Un appel unique « lis cette photo et explique » marche à peu près, mais :
- sur une photo de tableau, une partie du texte est petite ou floue → le modèle **devine** ;
- les LLM se trompent sur les exposants, les signes, les indices → l'explication part d'une formule fausse ;
- il n'a aucune mémoire des cours précédents.

Ici le modèle dispose d'outils et décide lui-même quand les utiliser :

| Outil | Rôle |
|---|---|
| `zoom_region` | relire une zone agrandie (image nettoyée ou photo brute) au lieu de deviner |
| `check_math` | vérifier une égalité, une dérivée, une primitive, résoudre une équation (sympy) |
| `search_course_notes` | chercher dans les leçons déjà digitalisées (mémoire long terme) |
| `save_lesson` | produire le résultat final dans un format structuré (schéma JSON strict) |

### Architecture

```
photo ──► preprocess.py ──► image redressée + nettoyée
                                   │
                                   ▼
                    ┌──────── agent.py (boucle) ────────┐
                    │  LLM (vision) ── appel d'outil ──► tools.py
                    │      ▲                              │  ├─ zoom_region   (preprocess)
                    │      └──────── résultat ◄───────────┘  ├─ check_math    (mathcheck.py / sympy)
                    │                                        ├─ search_notes  (memory.py)
                    └── arrêt quand save_lesson est appelé   └─ save_lesson   (memory.py)
                                   │
                                   ▼
                      leçon (en JSON) ──► export.py ──► Markdown
```

La boucle est écrite à la main (`eduboard/agent.py`) sans framework, pour garder le contrôle :
- nombre maximal de tours (`EDUBOARD_MAX_STEPS`) ;
- limite de 8 zooms par analyse (chaque image coûte des tokens) ;
- si le modèle s'arrête sans appeler `save_lesson`, on le relance **une** fois ;
- les erreurs d'outil sont renvoyées au modèle (`is_error`) pour qu'il corrige son appel ;
- chaque étape (réflexion, appel d'outil, résultat) est tracée et affichée dans l'interface.

### Prétraitement de l'image

`eduboard/preprocess.py` :
- **détection du tableau** : Canny + contours, on garde le plus grand quadrilatère convexe (> 20 % de l'image) ;
- **redressement** : transformation de perspective à partir des 4 coins ;
- **nettoyage** : on estime le fond (dilatation + flou médian, ce qui « efface » les traits fins) puis on divise l'image par ce fond → les ombres et reflets disparaissent, puis CLAHE pour le contraste ;
- **tableau noir/vert** : détecté par la luminance médiane, l'image est inversée avant nettoyage.

Si les bords ne sont pas trouvés (tableau qui dépasse de la photo), on garde l'image entière et on le signale au modèle.

## Installation

Python 3.10+.

```bash
cd ./eduboard-ai
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / Mac
pip install -r requirements.txt
copy .env.example .env        # puis mettre sa clé API dans .env
```

> Sous Windows avec Smart App Control, l'erreur `DLL load failed while importing jiter`
> vient des versions récentes de `jiter` (dépendance du SDK). `requirements.txt` fixe
> donc `jiter<0.18`.

## Utilisation

Interface web :
```bash
streamlit run app.py
```

Ligne de commande :
```bash
# juste le prétraitement (pas besoin de clé API)
python main.py clean photo.jpg -o propre.jpg

# analyse complète
python main.py analyse photo.jpg --course "Analyse 1" -o lecon.md

# leçons enregistrées
python main.py history --course "Analyse 1"
```

Pas de photo sous la main ? `python -m eduboard.synthetic samples/fake_board.jpg` génère un faux tableau photographié de biais.

## Tests

```bash
pytest -q
```

Les tests de l'agent utilisent un faux client qui rejoue un scénario (zoom → vérification → sauvegarde), donc ils tournent sans clé API et sans coût.

## Configuration (`.env`)

| Variable | Défaut | |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | clé API |
| `EDUBOARD_MODEL` | `claude-opus-5-5` | modèle (doit supporter la vision) |
| `EDUBOARD_EFFORT` | `medium` | `low` / `medium` / `high` : profondeur de raisonnement |
| `EDUBOARD_MAX_STEPS` | `12` | tours max de la boucle |
| `EDUBOARD_FALLBACK` | `1` | repli automatique sur un autre modèle si une requête est refusée |

## Limites connues

- L'écriture manuscrite très penchée ou très serrée reste difficile, même avec le zoom.
- La recherche dans la mémoire est par mots-clés (pas d'embeddings) : suffisant pour un semestre de cours, pas pour des milliers de documents.
- `check_math` ne lit pas le LaTeX, le modèle doit convertir en syntaxe sympy. Pas encore de timeout sur sympy.
- Les schémas sont seulement décrits, pas redessinés.

## Pistes d'amélioration

- Plusieurs photos pour un même cours (tableau effacé au fur et à mesure).
- Export PDF / Anki pour le quiz.
- Recherche sémantique dans la mémoire.
