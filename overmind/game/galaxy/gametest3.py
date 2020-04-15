import pygame, sys
import random
import math
import sys; sys.path.append('./pylsl')
from pylsl import StreamInlet, resolve_stream
from pygame.locals import*

pygame.init()

tela = pygame.display.set_mode((800, 600))
sair = False
clock = pygame.time.Clock()
space = pygame.image.load("space2.png")

#nave
##nave = pygame.image.load("nave.png")
x = 346
y = 480
morreu = pygame.mixer.Sound('ded.ogg')

#score
scorev = 0
scoreim = pygame.font.Font('freesansbold.ttf',32)
textx = 10
texty = 10

#asteroid
asteroidim = pygame.image.load("asteroid3.png")
asteroidx = random.choice([0,350])
asteroidy = -130

class navesp(pygame.sprite.Sprite)
    def _init_(self):
        pygame.sprite.Sprite._init_(self)
        self.naveim = pygame.image.load("nave.png")
        self.rect = naveim.get_rect()
        self.rect.centerx = x
        selfrect.centery = y
        tela.blit(self.naveim, self.rect)


def asteroid(x,y):
    asteroid.rect = asteroidim.get_rect()
    tela.blit(asteroidim, (asteroidx,asteroidy))

def show_score(x,y):
    score = scoreim.render("Score: " + str(scorev), True, (255,255,255))
    tela.blit(score, (textx,texty))
    

while sair != True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sair = True

        asteroidy += 3
        tela.fill((0,0,0))
        tela.blit(space, (0,0))
        

        



	##nave_mechanics
        ##nave = pygame.image.load("nave.png")

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x-=7	

            if event.key == pygame.K_RIGHT:

                x+=7

        if x <=0:
            x=0
        elif x >= 670:
            x = 670

	#asteroid_mechanics

        if asteroidy >=550:
            asteroidx=random.choice([0,350])
            asteroidy=-130

	#colisão
        distancex = math.pow(x-asteroidx,1)
        distancey = math.pow(y-asteroidy,1)

        if distancey <= 330 and -10 <= distancex <= 390:
            nave = pygame.image.load("explosion.png")
	    ##morreu.play()
        if nave.rect.colliderect(asteroid):
            print('colidiu')
			
	##Score
        if distancey == -23 and -10 >= distancex:
            scorev += 1
        if distancey == -23 and distancex >= 390:
            scorev += 1

        show_score(textx,texty)
        asteroid(x,y)
        player(x,y)
        clock.tick(33)
		
        pygame.display.update()

pygame.quit()
