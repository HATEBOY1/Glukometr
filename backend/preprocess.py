import cv2
import numpy as np


def upscale(img, min_side=120):
    h, w = img.shape[:2]
    short = min(h, w)
    if short >= min_side or short == 0:
        return img
    scale = max(1.0, min_side / short)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def to_gray_clahe(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize_adaptive(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def binarize_otsu(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def deskew(gray):
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 20:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_with_padding(img, box, pad=5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_for_ocr(crop_bgr, to_bgr=True):
    img = upscale(crop_bgr, min_side=120)
    gray = to_gray_clahe(img)
    gray = deskew(gray)
    bw = binarize_adaptive(gray)
    if to_bgr:
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return bw