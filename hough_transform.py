import cv2
import numpy as np
import math
from collections import defaultdict

# http://www.keymolen.com/2013/05/hough-transformation-c-implementation.html

class Accumulator:
    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)
        self.data = defaultdict(int)

    def increment(self, r, th):
        self.data[(int(r), int(th))] += 1

    def is_local_maximum(self, center_r, center_th):
        n = self.get(center_r, center_th)
        for dr in range(-4, 5):
            for dth in range(-4, 5):
                if self.get(center_r+dr, center_th+dth) > n:
                    return False
        return True

    def get(self, r, th):
        return self.data[(int(r), int(th))]


class HoughTransform:
    def __init__(self, image, threthold):
        self.image       = image
        self.accumulator = None
        self.threthold   = threthold

    def transform(self):
        img_h, img_w = self.image.shape
        center_x, center_y = img_w/2.0, img_h/2.0

        hough_h = math.sqrt(2.0) * max(img_w, img_h) / 2.0
        self.accumulator = Accumulator(180, hough_h * 2.0)

        for y in range(img_h):
            for x in range(img_w):
                if self.image[y, x] < 250:
                    continue
                
                for t in range(180):
                    r = (x - center_x) * math.cos(math.radians(t)) + (y - center_y) * math.sin(math.radians(t))
                    self.accumulator.increment(r + hough_h, t)

    def get_lines(self):
        if self.accumulator == None:
            return []

        img_h, img_w = self.image.shape
        accu_h = self.accumulator.h

        lines = []
        for r in range(self.accumulator.h):
            for t in range(self.accumulator.w):
                if self.accumulator.get(r, t) < self.threthold: continue
                if not self.accumulator.is_local_maximum(r, t): continue

                rad = math.radians(t)
                if 45 <= t and t <= 135:
                    # y = (r - x cos(t)) / sin(t)
                    x1 = 0
                    y1 = ((r - accu_h/2) - (x1 - img_w/2) * math.cos(rad)) / math.sin(rad) + img_h/2
                    x2 = img_w - 0
                    y2 = ((r - accu_h/2) - (x2 - img_w/2) * math.cos(rad)) / math.sin(rad) + img_h/2
                else:
                    # x = (r - y sin(t)) / cos(t)
                    y1 = 0
                    x1 = ((r - accu_h/2) - (y1 - img_h/2) * math.sin(rad)) / math.cos(rad) + img_w/2
                    y2 = img_h - 0
                    x2 = ((r - accu_h/2) - (y2 - img_h/2) * math.sin(rad)) / math.cos(rad) + img_w/2
                lines.append(((int(x1), int(y1)), (int(x2), int(y2))))
        return lines

