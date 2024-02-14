import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

#1. Compressão de imagens bmp no formato jpeg utilizando um editor de imagem (e.g., GIMP).
#1.1. Comprima as imagens fornecidas segundo o codec JPEG, com qualidade alta (Q=75).
#1.2. Comprima as imagens fornecidas segundo o codec JPEG, com qualidade média (Q=50).
#1.3. Comprima as imagens fornecidas segundo o codec JPEG, com qualidade baixa (Q=25).
#1.4. Compare os resultados e tire conclusões.
#Conclusão: Quanto maior for a taxa de compressão pior será a qualidade da imagem, sendo então o ideal encontrar
#um meio termo que permita ter um certo nível de compressão de imagem e que não comprometa a sua qualidade

#2: criar o encoder e decoder
def encoder(img):
    R, G, B = splitRGB(img)
    return R,G,B

def decoder(R,G,B):
    imgRec = joinRGB(R, G, B)
    return imgRec

#3.2 Crie uma função para implementar um colormap definido pelo utilizador.
def newCmap(keyColors = [(0,0,0),(1,1,1)], name = "gray", N= 256):
    cm = clr.LinearSegmentedColormap.from_list(name, keyColors, N)
    return cm

#3.3 Crie uma função que permita visualizar a imagem com um dado colormap.
def showImg(img, fname="", caption="", cmap=None):
    print(img.shape)  # Imprime as dimensões da imagem
    print(img.dtype)  # Imprime o tipo de dados da imagem
    plt.figure()  # Cria uma nova figura
    plt.imshow(img, cmap)  # Mostra a imagem com o mapa de cores aplicado
    plt.axis('off')  # Remove os eixos
    plt.title(caption + fname)  # Define o título da imagem
    plt.show()  # Exibe a imagem

#3.4. Encoder: Crie uma função para separar a imagem nos seus componentes RGB.
def splitRGB(img):
    R = img[:, :, 0]  # Extrai o canal vermelho
    G = img[:, :, 1]  # Extrai o canal verde
    B = img[:, :, 2]  # Extrai o canal azul
    return R, G, B

#3.5. Decoder: Crie também a função inversa (que combine os 3 componentes RGB).
def joinRGB(R,G,B):
    nl, nc = R.shape  # Obtém as dimensões da imagem a partir do canal vermelho
    imgRec = np.zeros((nl, nc, 3), dtype=np.uint8)  # Cria uma nova imagem vazia
    imgRec[:, :, 0] = R  # Define o canal vermelho
    imgRec[:, :, 1] = G  # Define o canal verde
    imgRec[:, :, 2] = B  # Define o canal azul
    return imgRec 

def main():
    # 3.1 Leia uma imagem .bmp, e.g., a imagem peppers.bmp.
    fname = "airport.bmp"
    img = plt.imread(fname)
   
    #3.2 Crie uma função para implementar um colormap definido pelo utilizador.
    cm_red=newCmap([(0,0,0),(1,0,0)], "cm_red", 256)
    cm_green=newCmap([(0,0,0),(0,1,0)], "cm_green", 256)
    cm_blue=newCmap([(0,0,0),(0,0,1)], "cm_blue", 256)
    cm_gray=newCmap([(0,0,0),(1,1,1)], "cm_gray", 256)

    #3.3 Crie uma função que permita visualizar a imagem com um dado colormap.
    showImg(img,fname,"Imagem original: ")
    
    #3.4 Encoder: Crie uma função para separar a imagem nos seus componentes RGB.
    R,G,B = encoder(img)

    #3.5 Decoder: Crie também a função inversa (que combine os 3 componentes RGB).
    imgRec = decoder(R,G,B)

    #3.6 Visualize a imagem e cada um dos canais RGB (com o colormap adequado).
    showImg(R,fname,"Img Red: ",cm_red)
    showImg(G,fname,"Img Green: ",cm_green)
    showImg(B,fname,"Img Blue: ",cm_blue)
    
    return

if __name__ == "__main__":
    main()