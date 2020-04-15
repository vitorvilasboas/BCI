import pygame, sys
from pygame.locals import*



def main():
    pygame.init()
    tela = pygame.display.set_mode((900, 600))
    sair = False
    clock = pygame.time.Clock()
    branco = (255,255,255)
    azul = (108,194,236)
    verde = (152,231,114)
    sup = pygame.Surface((300,300))
    space = pygame.image.load("space.png")
    nave = pygame.image.load("nave.png")
    x = 150
    y = 155


    
    
    while sair != True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sair = True

        if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x-=5
                    

            
                if event.key == pygame.K_RIGHT:
                    x+=5


        
        tela.blit(nave,[x,y])
        
        clock.tick(27)
        ##tela.blit(space,[1,1])
        pygame.display.update()

    pygame.quit()
            
        
    
main()
