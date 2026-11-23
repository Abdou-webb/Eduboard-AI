"""Traitement de la photo du tableau avant de l'envoyer au modele.

Etapes : detection du tableau -> redressement (perspective) -> nettoyage
(on enleve les reflets / ombres en estimant le fond).
"""
import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 1600  # au-dela ca coute des tokens pour rien


@dataclass
class BoardScan:
    original: np.ndarray
    warped: np.ndarray
    clean: np.ndarray
    corners: np.ndarray | None = None
    inverted: bool = False  # True si tableau noir / vert

    @property
    def board_found(self) -> bool:
        return self.corners is not None


def load_image(src) -> np.ndarray:
    if isinstance(src, (bytes, bytearray)):
        buf = np.frombuffer(src, np.uint8)
    else:
        # cv2.imread plante sur les chemins avec accents sous Windows
        buf = np.fromfile(str(Path(src)), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Impossible de lire l'image")
    return img


def resize_max(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Remet les 4 coins dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype="float32",
    )


def find_board(img: np.ndarray, min_area_ratio: float = 0.2) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area:
            break
        if area > 0.98 * img_area:
            # c'est le bord de la photo, pas le tableau
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx)
    return None


def warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (w, h))


def is_dark_board(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray)) < 110


def clean_board(img: np.ndarray) -> np.ndarray:
    # Idee : le fond du tableau varie lentement (ombres, reflets du neon),
    # les traits de feutre sont fins. On estime le fond en "effacant" les
    # traits (dilatation + flou median) puis on divise l'image par ce fond.
    out = np.empty_like(img)
    kernel = np.ones((7, 7), np.uint8)
    for i in range(3):
        ch = img[:, :, i]
        bg = cv2.medianBlur(cv2.dilate(ch, kernel), 21)
        out[:, :, i] = cv2.divide(ch, bg, scale=255)

    # petit coup de CLAHE sur la luminance pour faire ressortir les traits pales
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def scan(src) -> BoardScan:
    original = resize_max(load_image(src) if not isinstance(src, np.ndarray) else src)

    corners = find_board(original)
    warped = warp(original, corners) if corners is not None else original.copy()

    inverted = is_dark_board(warped)
    base = 255 - warped if inverted else warped  # craie blanche -> traits sombres

    return BoardScan(
        original=original,
        warped=warped,
        clean=clean_board(base),
        corners=corners,
        inverted=inverted,
    )


def crop_normalized(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Decoupe une zone donnee en coordonnees relatives (0..1)."""
    H, W = img.shape[:2]
    x0 = int(np.clip(x, 0, 1) * W)
    y0 = int(np.clip(y, 0, 1) * H)
    x1 = int(np.clip(x + w, 0, 1) * W)
    y1 = int(np.clip(y + h, 0, 1) * H)
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError("Zone trop petite")
    crop = img[y0:y1, x0:x1]
    # on agrandit les petites zones sinon le zoom ne sert a rien
    if max(crop.shape[:2]) < 600:
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return crop


def to_jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("encodage jpeg rate")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def to_rgb(img: np.ndarray) -> np.ndarray:
    # pour streamlit / PIL
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
