import cv2
import numpy as np


def put_render_on_image(render, image):
    render_bgr = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
    lower_white = np.array([252, 252, 252])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(render_bgr, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel)

    mask_inv = cv2.bitwise_not(mask)

    image_bg = cv2.bitwise_and(image, image , mask=mask)
    image_fg = cv2.bitwise_and(render_bgr, render_bgr, mask=mask_inv)
    dst = cv2.add(image_fg, image_bg)

    cv2.imshow("render", render_bgr)
    #cv2.imshow("mask", mask)
    #cv2.imshow("mask inv", mask_inv)
    #cv2.imshow("image foregroung", image_fg)
    #cv2.imshow("image background", image_bg)
    #cv2.imshow("dst", dst)

    return dst
