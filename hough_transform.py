import cv2
import numpy as np
import math
import pickle
from collections import defaultdict

from abc import ABCMeta, abstractmethod

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


class Accumulator3D:
    def __init__(self, x_size, y_size, r_size):
        self.x_size = int(x_size)
        self.y_size = int(y_size)
        self.r_size = int(r_size)
        self.data = np.zeros((r_size, y_size, x_size))

    def increment(self, x, y, r):
        self.data[(int(r), int(y), int(x))] += 1

    def is_local_maximum(self, cx, cy, cr):
        n = self.get(cx, cy, cr)
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                for dr in range(-4, 5):
                    if 0 <= cx + dx and cx + dx < self.x_size and \
                       0 <= cy + dy and cy + dy < self.y_size and \
                       0 <= cr + dr and dr + dr < self.r_size:
                        if self.data[cr+dr, cy+dy, cx+dx] > n:
                            return False
        return True

    def get(self, x, y, r):
        return self.data[(int(r), int(y), int(x))]
    
    def indexes(self):
        for x, y, r in np.ndindex(self.x_size, self.y_size, self.r_size):
            yield x, y, r


class HoughTransform(metaclass=ABCMeta):
    def __init__(self, image):
        self.image       = image
        self.accumulator = None

    @abstractmethod
    def hough_transform(self):
        pass

    @abstractmethod
    def detect(self, threshold):
        pass


class LineDetector(HoughTransform):
    def hough_transform(self):
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

    def detect(self, threshold):
        if self.accumulator == None:
            return []

        img_h, img_w = self.image.shape
        accu_h = self.accumulator.h

        lines = []
        for r in range(self.accumulator.h):
            for t in range(self.accumulator.w):
                if self.accumulator.get(r, t) < threshold: continue
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


class CircleDetector(HoughTransform):
    RMAX = 1000

    def hough_transform(self):
        img_h, img_w = self.image.shape
        self.accumulator = Accumulator3D(img_w, img_h, self.RMAX+1)

        for y in range(img_h):
            for x in range(img_w):
                if self.image[y, x] < 250:
                    continue

                for cy in range(img_h):
                    for cx in range(img_w):
                        r = (cy - y) ** 2 + (cx - x) ** 2
                        if r > self.RMAX: continue

                        self.accumulator.increment(cx, cy, r)

    def detect(self, threshold):
        if self.accumulator == None:
            return []

        img_h, img_w = self.image.shape
        r_size = self.accumulator.r_size

        circles = []
        for x, y, r in self.accumulator.indexes():
            if self.accumulator.get(x, y, r) < threshold: continue
            if not self.accumulator.is_local_maximum(x, y, r): continue

            circles.append({"cx": int(x), "cy": int(y), "radius": int(math.sqrt(r))})

        return circles

    def save_accumulator(self, file_name="accumulator.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.accumulator, f)

    def load_accumulator(self, file_name="accumulator.pkl"):
        with open(file_name, "rb") as f:
            self.accumulator = pickle.load(f)
