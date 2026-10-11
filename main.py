
import cv2
import numpy as np
from pathlib import Path


def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError("Image could not be loaded.")

    return image


def detect_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours[:10]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if len(approx) == 4:
            return approx.reshape(4, 2), edges

    return None, edges


def enhance_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )


def scan_document(input_path, output_path="scanned_document.jpg"):
    image = load_image(input_path)
    corners, edges = detect_document(image)

    if corners is None:
        print("Document boundary not detected.")
        print("Saving an enhanced version of the full image.")

        result = enhance_document(image)
    else:
        points = corners.astype(np.float32)

        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).ravel()

        tl = points[np.argmin(sums)]
        br = points[np.argmax(sums)]
        tr = points[np.argmin(diffs)]
        bl = points[np.argmax(diffs)]

        width = int(max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl)
        ))
        height = int(max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl)
        ))

        if width < 2 or height < 2:
            raise ValueError("Invalid document dimensions.")

        destination = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(
            np.array([tl, tr, br, bl], dtype=np.float32),
            destination
        )

        warped = cv2.warpPerspective(
            image, matrix, (width, height)
        )

        result = enhance_document(warped)

    cv2.imwrite(str(output_path), result)
    print(f"Scanned document saved to: {output_path}")


if __name__ == "__main__":
    print("Smart Document Scanner")
    print("Use scan_document('your_image.jpg') to scan an image.")
