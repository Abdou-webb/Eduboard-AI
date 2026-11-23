"""Genere une fausse photo de tableau (utile pour les tests et les demos
quand on n'a pas de vraie photo sous la main)."""
import cv2
import numpy as np

LINES = [
    ("Derivee d'un produit", (0, 0, 180)),
    ("(u.v)' = u'v + uv'", (30, 30, 30)),
    ("Ex: f(x) = x^2 sin(x)", (30, 30, 30)),
    ("f'(x) = 2x sin(x) + x^2 cos(x)", (160, 60, 0)),
]


def make_fake_board(seed: int = 0, dark: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)

    board_w, board_h = 900, 560
    bg_color = (40, 70, 45) if dark else (245, 245, 240)
    board = np.full((board_h, board_w, 3), bg_color, np.uint8)

    y = 90
    for text, color in LINES:
        if dark:
            color = (235, 235, 235)
        cv2.putText(board, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 3, cv2.LINE_AA)
        y += 110
    cv2.rectangle(board, (0, 0), (board_w - 1, board_h - 1), (90, 90, 90), 10)

    # le "mur" derriere
    scene = np.full((900, 1300, 3), (120, 140, 160), np.uint8)
    noise = rng.integers(-12, 12, scene.shape, dtype=np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # perspective : on simule une photo prise de biais
    src = np.float32([[0, 0], [board_w, 0], [board_w, board_h], [0, board_h]])
    dst = np.float32([[230, 160], [1120, 210], [1080, 760], [190, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, M, (scene.shape[1], scene.shape[0]))
    mask = cv2.warpPerspective(np.full((board_h, board_w), 255, np.uint8), M, (scene.shape[1], scene.shape[0]))
    scene[mask > 0] = warped[mask > 0]

    # ombre / eclairage non uniforme (gradient horizontal)
    grad = np.linspace(0.65, 1.05, scene.shape[1])[None, :, None]
    scene = np.clip(scene * grad, 0, 255).astype(np.uint8)
    return scene


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "samples/fake_board.jpg"
    cv2.imwrite(out, make_fake_board())
    print("ecrit :", out)
