import pygame, sys
from pygame.locals import*

class ship(pygame.sprite.Sprite):
        def _init_(self):
            pygame.sprite.Sprite._init_(self)
            self.imgs = pygame.image.load("ship.png")

            self.rect = self.imgs.get_rect()
            self.rect.centerx = 150
            self.rect.centery = -50

        def colocar (self, superficie):
            superficie.blit(self.imgs, self.rect)
def main():
    pygame.init()
    tela = pygame.display.set_mode([900, 600])
    sair = False
    clock = pygame.time.Clock()
    branco = (255,255,255)
    azul = (108,194,236)
    verde = (152,231,114)
    sup = pygame.Surface((300,300))
    x = 150
    y = 150
    space = pygame.image.load("space.png")
    jogador = ship()
        

    while sair != True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sair = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    ##ret.move(200,100)
                    x-=3
                    
                    print('left');
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    x+=3
                    print('right');
            ret = pygame.Rect(x, y, 50, 100)
            jogador.colocar(tela)


       
        
        clock.tick(27)
        tela.blit(space,[1,1])
        pygame.draw.rect(tela,verde,ret)
        ##tela.fill((0,0,0))
        pygame.display.update()

    pygame.quit()
            
        
    
main()
