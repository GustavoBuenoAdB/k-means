import sys
import time
import random
import numpy as np
import cv2
import math

#===============================================================================

INPUT_IMAGE =  "teste5.bmp"

TAM_QUAD_VISIVEL = 800
TAM_QUAD_GRID = 4

VISUALISA = False
AUMENTA = False
N_AUMENTA = 2

N_CORES = 2
MAX_INTERACOES = 10

def dist_euclidiana(x1, y1, z1, x2, y2, z2):
    return math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2) + ((z2 - z1)**2))

def k_means(img, n):

    #definindo os pontos aleatorios do k-means
    cores = np.zeros((n,3))

    for i in range(n):
        cores[i][:] = img[random.randint(0, img.shape[0] - 1)][random.randint(0, img.shape[1] - 1)][:]

    #mapa pra saber a separacao dos grupos inicializado em um grupo que nao existe
    mapa = np.zeros((TAM_QUAD_GRID, TAM_QUAD_GRID), dtype = np.int8)

    moveu = True
    interacoes = 0
    somas = np.zeros((n, 3 , 2)) # n = numero de grupos, 3 = num de canais, 2 = a soma geral e o numero de elementos somados
    for z in range(3):
        somas[0][z][0] += np.sum(img[:,:,z])
        somas[0][z][1] += (img.shape[0] * img.shape[1]) 

    while moveu and interacoes < MAX_INTERACOES:
        moveu = False
        interacoes += 1
        
        #escolhendo a cor mais perto pra cada ponto
        for y in range(TAM_QUAD_GRID):
            for x in range(TAM_QUAD_GRID):
                #inicializa no valor do antigo grupo como menor distancia
                min = dist_euclidiana(img[y][x][0], img[y][x][1], img[y][x][2], cores[mapa[y][x]][0], cores[mapa[y][x]][1], cores[mapa[y][x]][2])
                grupo = mapa[y][x]
                
                for i in range(n):
                    dist = dist_euclidiana(img[y][x][0], img[y][x][1], img[y][x][2], cores[i][0], cores[i][1], cores[i][2])
                    if (dist < min):
                        min = dist
                        grupo = i
                        moveu = True

                        #desconta do grupo anterior
                        for z in range(3):
                            somas[mapa[y][x]][z][0] -= img[y][x][z]
                            somas[mapa[y][x]][z][1] -= 1
                        
                        mapa[y][x] = grupo

                        #adiciona no novo grupo
                        for z in range(3):
                            somas[mapa[y][x]][z][0] += img[y][x][z]
                            somas[mapa[y][x]][z][1] += 1
        
        #Recalculando as cores, as centroides
        for i in range(n):
            for z in range(3):
                if somas[i][z][1] > 0:
                    cores[i][z] = somas[i][z][0] / somas[i][z][1]
                else:
                    cores[i][:] = img[random.randint(0, img.shape[0] - 1)][random.randint(0, img.shape[1] - 1)][:]

    # quando terminado trocando as cores aintigas pelas novas na imagem
    for y in range(TAM_QUAD_GRID):
        for x in range(TAM_QUAD_GRID):
                img[y][x][:] = cores[mapa[y][x]][:]
    
    return img


def main():
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    
    #convertendo em float
    img = img.astype(np.float32) / 255


    img_grid = np.zeros((TAM_QUAD_GRID, TAM_QUAD_GRID, 3), dtype=np.float32)
    img_saida = np.zeros_like(img)

    inicio = time.perf_counter()
    #percorrendo a imagem com a janela saltitante das Grids
    for y in range(0, img.shape[0] - TAM_QUAD_GRID + 1, TAM_QUAD_GRID):
        for x in range(0, img.shape[1] - TAM_QUAD_GRID + 1, TAM_QUAD_GRID):
            img_grid[:,:,:] = img[y : y + TAM_QUAD_GRID , x : x + TAM_QUAD_GRID , :]
            
            if VISUALISA:
                resized_img = cv2.resize(img_grid, (TAM_QUAD_VISIVEL, TAM_QUAD_VISIVEL), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('img', resized_img)
                cv2.waitKey()
                cv2.destroyAllWindows()
            
            #chamando o k_means para cada grid interada
            img_grid = k_means(img_grid, N_CORES)

            img_saida[y : y + TAM_QUAD_GRID , x : x + TAM_QUAD_GRID , :] = img_grid[:,:,:]

            if VISUALISA:
                resized_img = cv2.resize(img_grid, (TAM_QUAD_VISIVEL, TAM_QUAD_VISIVEL), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('img', resized_img)
                cv2.waitKey()
                cv2.destroyAllWindows()


    fim = time.perf_counter()
    print("tempo: " + str(fim - inicio))
    if AUMENTA:
        img = cv2.resize(img, (img.shape[1] * N_AUMENTA , img.shape[0] * N_AUMENTA), interpolation=cv2.INTER_NEAREST)
        img_saida = cv2.resize(img_saida, (img_saida.shape[1] * N_AUMENTA , img_saida.shape[0] * N_AUMENTA), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)
    cv2.imshow('img_k_means', img_saida)
    cv2.waitKey()
    cv2.destroyAllWindows()
                

    

if __name__ == '__main__':
    main()