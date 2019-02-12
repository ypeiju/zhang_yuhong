
import cv2 as cv
import numpy as np

image = cv.imread("animal.jpg")

# sharpen
kernel = np.array([
    [0, -2, 0],
    [-2, 9, -2],
    [0, -2, 0]])

sharpen = cv.filter2D(image, -1, kernel)
cv.imwrite('sharpen.jpg', sharpen)

#deges

kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]])

edges = cv.filter2D(image, -1, kernel)
cv.imwrite('edges.jpg', edges)

#blur
kernel = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [1, 1, 1]]) / 9.0

blur = cv.filter2D(image, -1, kernel)
cv.imwrite('blur.jpg', blur)

#emboss
kernel = np.array([
    [-2, -1,0],
    [-1, 1, 1],
    [0, 1, 2]])

emboss = cv.filter2D(image, -1, kernel)
emboss = cv.cvtColor(emboss, cv.COLOR_BGR2GRAY)
cv.imwrite('emboss.jpg', emboss)

cv.destroyAllWindows() 