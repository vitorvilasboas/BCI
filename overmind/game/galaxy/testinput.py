import pygame
import sys
import pyautogui

pygame.init()
msg = input
cd = 0
while cd <= 1:
        cd +=1
        pyautogui.typewrite('takeoff', interval = 0.5)
        pyautogui.press('enter')
    
