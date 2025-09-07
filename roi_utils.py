import cv2


def detect_eye_region(img):
    """Haar cascade 기반의 간단한 눈 영역 검출. 실패 시 전체 이미지 반환."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_eye_tree_eyeglasses.xml")
    eyes = cascade.detectMultiScale(gray, 1.1, 3)
    if len(eyes) == 0:
        h, w = img.shape[:2]
        return 0, 0, w, h
    x, y, w, h = max(eyes, key=lambda e: e[2] * e[3])
    pad = int(0.1 * w)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    return x1, y1, x2, y2
