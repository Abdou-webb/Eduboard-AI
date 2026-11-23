import numpy as np
import pytest

from eduboard.preprocess import crop_normalized, find_board, order_corners, scan
from eduboard.synthetic import make_fake_board


def test_order_corners():
    pts = np.array([[100, 90], [0, 0], [0, 80], [110, 5]])
    tl, tr, br, bl = order_corners(pts)
    assert tuple(tl) == (0, 0)
    assert tuple(tr) == (110, 5)
    assert tuple(br) == (100, 90)
    assert tuple(bl) == (0, 80)


def test_board_is_detected_and_warped():
    img = make_fake_board()
    corners = find_board(img)
    assert corners is not None
    # les coins doivent etre proches de ceux utilises pour generer l'image
    expected = np.float32([[230, 160], [1120, 210], [1080, 760], [190, 700]])
    assert np.abs(corners - expected).max() < 25

    sc = scan(img)
    h, w = sc.warped.shape[:2]
    assert 1.4 < w / h < 1.9  # le tableau d'origine fait 900x560 (~1.6)


def test_cleaning_flattens_background():
    sc = scan(make_fake_board())
    gray = sc.clean.mean(axis=2)
    # bande gauche vs bande droite : l'ombre du generateur doit avoir disparu
    left, right = gray[:, 20:80], gray[:, -80:-20]
    assert abs(np.median(left) - np.median(right)) < 15


def test_dark_board_is_inverted():
    sc = scan(make_fake_board(dark=True))
    assert sc.inverted
    assert np.median(sc.clean) > 150  # fond clair apres inversion


def test_no_board_found_on_flat_image():
    img = np.full((400, 600, 3), 200, np.uint8)
    sc = scan(img)
    assert not sc.board_found
    assert sc.warped.shape == img.shape


def test_crop_normalized():
    img = np.zeros((500, 1000, 3), np.uint8)
    crop = crop_normalized(img, 0.5, 0.5, 0.2, 0.2)
    assert crop.shape[:2] == (200, 400)  # agrandi x2 car petit
    with pytest.raises(ValueError):
        crop_normalized(img, 0.99, 0.99, 0.5, 0.5)
