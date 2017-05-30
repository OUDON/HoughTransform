import cv2
import numpy as np
from hough_transform import LineDetector, CircleDetector

def preprocess(img):
    img_blur = cv2.blur(img, (5, 5))            # Smoothing
    img_edge = cv2.Canny(img_blur, 100, 150, 3) # Edge detection
    return img_edge


def detect_lines():
    FILE_PATH = "image/farm.png"
    img_color = cv2.imread(FILE_PATH)
    img_preprocessed = preprocess(img_color)

    h, w = img_preprocessed.shape

    cv2.imshow("Preprocessed", img_preprocessed)
    cv2.waitKey(0)

    print("Image cols: {}".format(w))
    print("Image rows: {}".format(h))

    detector = LineDetector(img_preprocessed)
    detector.hough_transform()
    lines = detector.detect(175)
    print("Lines: {}".format(lines))

    img_res = img_color.copy()
    for l in lines:
        cv2.line(img_res, l[0], l[1], (0, 0, 255), 2, 8)

    cv2.imshow("Result", img_res)
    cv2.waitKey(0)


def detect_circles():
    FILE_PATH = "image/cd.jpg"
    img_color = cv2.imread(FILE_PATH)
    img_preprocessed = preprocess(img_color)

    h, w = img_preprocessed.shape

    detector = CircleDetector(img_preprocessed)

    try:
        detector.load_accumulator("accumlator_circle.pkl")
    except:
        print("Begin Hough Transform ...")
        detector.hough_transform()
        detector.save_accumulator("accumlator_circle.pkl")

    print("Detecting ...")
    circles = detector.detect(10)

    print("Circles: {}".format(circles))
    img_res = img_color.copy()
    for c in circles:
        cv2.circle(img_res, (c["cx"], c["cy"]), c["radius"], (0, 0, 255), 2, 8)
    cv2.imshow("Result", img_res)
    cv2.waitKey(0)


def main():
    # detect_lines()
    detect_circles()


if __name__ == "__main__":
    main()
