import cv2
from hough_transform import HoughTransform

def preprocess(img):
    img_blur = cv2.blur(img, (5, 5))            # Smoothing
    img_edge = cv2.Canny(img_blur, 100, 150, 3) # Edge detection
    return img_edge


def main():
    FILE_PATH = "image/farm.png"
    img_color = cv2.imread(FILE_PATH)
    img_preprocessed = preprocess(img_color)

    h, w = img_preprocessed.shape

    print("Image cols: {}".format(w))
    print("Image rows: {}".format(h))

    hough = HoughTransform(img_preprocessed, 175)
    hough.transform()
    lines = hough.get_lines()
    print("Lines: {}".format(lines))

    img_res = img_color.copy()
    for l in lines:
        cv2.line(img_res, l[0], l[1], (0, 0, 255), 2, 8)

    cv2.imshow("Result", img_res)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
