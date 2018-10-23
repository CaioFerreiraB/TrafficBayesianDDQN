from skimage import io
from skimage.color import rgb2gray

import os


screenshot = io.imread(os.getcwd() + "/initial_screenshot.png")
print('shape original: ', screenshot.shape)

screenshot_grey = rgb2gray(screenshot)
print('shape gray: ', screenshot_grey.shape)

io.imsave(os.getcwd() + "/initial_screenshot_grey.png", screenshot_grey)

teste = io.imread(os.getcwd() + "/initial_screenshot_grey.png")
print(teste.shape)