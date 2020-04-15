import pygame, sys
import random
import math
from pygame.locals import*
import time
pygame.init()

tela = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
space = pygame.image.load("space2.png")

pygame.display.set_caption('teste')

#pygame.display.update() 

#Inicial
default_font = pygame.font.Font('freesansbold.ttf',32)
instruc = pygame.font.Font('freesansbold.ttf',32)
instruc2 = pygame.font.Font('freesansbold.ttf',32)
ready = pygame.font.Font('freesansbold.ttf',32)
go = pygame.font.Font('freesansbold.ttf',32)
inst = pygame.image.load("instruc2.png")
spacet = pygame.mixer.music.load('spacet.ogg')
#pygame.mixer.music.play()
startim = pygame.font.Font('freesansbold.ttf',32)

#nave
nave = pygame.image.load("nave.png")
x = 346 # largura
y = 480 # altura
move = pygame.mixer.Sound('woosh.ogg')
ded = pygame.mixer.Sound('ded.ogg')

#colisãoscore
colisoes = 0
colidx = 300
colidy = 100
colidim = pygame.font.Font('freesansbold.ttf',32)

#Tempo
timev = 0
timeim = pygame.font.Font('freesansbold.ttf',32)
timex = 300
timey = 300



#score
batida = 0
scorev = 0
scoreim = pygame.font.Font('freesansbold.ttf',32)
textx = 300
texty = 200



#asteroid
asteroidim = pygame.image.load("asteroid3.png")
##asteroidx = random.randint(0,5)
##asteroidy = random.randint(-150,-130)
asteroidx = random.choice([0,350])
asteroidy = -130
astemusic = pygame.mixer.Sound('asteroid.ogg')

#blit() - adiciona na superficie da tela

def player(x,y):
    tela.blit(nave, (x,y))

def asteroid(x,y):
    tela.blit(asteroidim, (asteroidx,asteroidy))

def show_score(x,y):
    score = scoreim.render("Score: " + str(scorev), True, (255,255,255))
    tela.blit(score, (x,y))
def start_button(x,y):
    startb = startim.render("Aperte Espaço para Iniciar",True,(255,255,255,))
    tela.blit(startb,(350,550))
def instrucoes(x,y): 
    instrucim = default_font.render("Desvie dos Asteroides",True,(255,255,255,))
    instrucim2 = default_font.render("para marcar pontos",True,(255,255,255,))
    tela.blit(instrucim, (50, 50))
    tela.blit(instrucim2, (50, 100))
    tela.blit(inst, (50,200))
def ready(x,y):
    readyim = instruc.render("READY",True,(255,255,255,))
    tela.blit(readyim, (350, 250))

    
def timer(x,y):
    time = timeim.render("Tempo: " + str(math.trunc(timev/33)), True, (255,255,255))
    tela.blit(time, (timex,timey))

def colid(x,y):
    colider = colidim.render("Asteroides: " + str(colisoes), True, (255,255,255))
    tela.blit(colider, (colidx,colidy))

sair = False 
play = False  # partida já iniciada?
game_over = False # fim de partida ?

fps = 33 # controla o tempo de jogo também

while not sair:
        for event in pygame.event.get(): # captura todos os eventos que ocorrem na tela do jogo
            #print(event)
            if event.type == pygame.QUIT:
                sair = True
        
        ## Tela inicial
        if play == False and game_over==False:
            tela.blit(space,(0,0)) #imagem de fundo
            instrucoes(x,y)
            start_button(x,y)
            pygame.display.update()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    play = True

        if game_over == True and play == False:
            tela.fill((0,0,0)) # preenche com uma cor sólida 
            show_score(textx,texty)
            timer(x,y)
            colid(x,y)
            clock.tick(fps) # limita os frames por segundo
            pygame.display.update() # atualiza tela
                   
        if game_over == False and play == True:
            nave = pygame.image.load("nave.png")
            timev += 1
            timev2 = math.trunc(timev/fps)
            print(timev,timev2)
            tela.fill((0,0,0))
            tela.blit(space, (0,0))
            if timev2<=1: ready(x,y)   
            if timev2 >= 3: asteroidy += 2
            
            # nave_mechanics
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    x-=7
                    move.play()

                if event.key == pygame.K_d:
                    x+=7
                    move.play()

            if x <=0: x=0
            elif x >= 670: x=670

            # asteroid_mechanics
            if asteroidy >=550:
                colisoes += 1
                asteroidx=random.choice([0,350])
                astemusic.play()
                asteroidy=-130
                
         	# colisão
            distancex = math.pow(x-asteroidx,1)
            distancey = math.pow(y-asteroidy,1)

            if distancey <= 330 and -9 <= distancex <= 389:
                nave = pygame.image.load("explosion.png")
                x = asteroidx + 230
                
            if timev2 == 60:
                game_over = True
                play =  False

            # Score
            if distancey == -22 and -9 >= distancex: scorev += 1
            if distancey == -22 and distancex >= 389: scorev += 1

            colid(x,y)
            show_score(x,y)
            
            asteroid(x,y)
            player(x,y)
            clock.tick(33)
            pygame.display.update()

pygame.quit()
