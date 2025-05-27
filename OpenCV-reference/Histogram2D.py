import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

iteration = 1
time_list = list()

image_bgr = cv2.imread('obraz.tif')
if image_bgr is None:
    raise FileNotFoundError("Nie znaleziono pliku 'obraz.tif'.")
end = time.time()


for i in range(iteration):
    start = time.time()

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    end = time.time()
    time_list.append(end - start)

print(f"Minimalny czas wykonania: {np.min(time_list) * 1000}, ms")
print(f"Maksymalny czas wykonania: {np.max(time_list) * 1000}, ms")
print(f"Średni czas wykonania: {np.mean(time_list) * 1000}, ms")


