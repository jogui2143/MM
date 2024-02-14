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
def encoder(img, pad = False, split = False):

  if split:
     R, G, B = splitRGB(img)
     return R,G,B

  elif pad:
    return padding(img)

def decoder(R,G,B,padded_img = None, og = None, unpad = False,join = False):

  if join:
     imgRec = joinRGB(R, G, B)
     return imgRec

  elif unpad:
    return unpadding(padded_img, og)


#3.2 Crie uma função para implementar um colormap definido pelo utilizador.
def newCmap(keyColors = [(0,0,0),(1,1,1)], name = "gray", N= 256):
    cm = clr.LinearSegmentedColormap.from_list(name, keyColors, N)
    return cm

#3.3 Crie uma função que permita visualizar a imagem com um dado colormap.
def showImg(img, fname="", caption="", cmap=None):
    #print(img.dtype)  # Imprime o tipo de dados da imagem
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

#4.1. Encoder: Crie uma função para fazer padding dos canais RGB. 
'''''
Obs: Caso a dimensão da imagem não seja múltipla de 32x32, faça padding da mesma, replicando a última linha
e a última coluna em conformidade.
'''''
def padding(img):
  # Captura a altura (h) e a largura (ln) da imagem.
  h,w = img.shape[:2]
  p1, p2, p3 = splitRGB(img)

  # Obtém as dimensões do primeiro canal de cor.
  r,c = p1.shape

  # Calcula quantos pixels faltam para a altura e a largura para serem múltiplos de 32.
  # Se a dimensão já for múltiplo de 32, não adiciona nenhum pixel (0).
  v_pad = 32 - (r % 32) if r % 32 > 0 else 0
  h_pad = 32 - (c % 32) if c % 32 > 0 else 0  

  # Adiciona pixels à última linha (replica a última linha) e à última coluna (replica a última coluna)
  # para fazer com que as dimensões sejam múltiplas de 32.
  # Faz isso para os três canais RGB separadamente.
  p1 = np.vstack([p1, np.repeat(np.array(p1[-1,:], ndmin = 2), v_pad, axis=0)])
  p1 = np.hstack([p1, np.repeat(np.array(p1[:,-1], ndmin = 2), h_pad, axis=0).T])

  p2 = np.vstack([p2, np.repeat(np.array(p2[-1,:], ndmin = 2), v_pad, axis=0)])
  p2 = np.hstack([p2, np.repeat(np.array(p2[:,-1], ndmin = 2), h_pad, axis=0).T])

  p3 = np.vstack([p3, np.repeat(np.array(p3[-1,:], ndmin = 2), v_pad, axis=0)])
  p3 = np.hstack([p3, np.repeat(np.array(p3[:,-1], ndmin = 2), h_pad, axis=0).T])

  # Empilha os três canais de volta em uma imagem e retorna junto com as dimensões originais.
  return np.dstack((p1, p2, p3)), (h,w)


#4.2. Decoder: Crie também a função inversa para remover o padding. 
'''''
Obs: Certifique-se de que recupera os canais RGB com a dimensão original, visualizando a imagem original.
'''''
def unpadding(img, og):
  return img[:og[0], :og[1], :]

def main():
    # 3.1 Leia uma imagem .bmp, e.g., a imagem peppers.bmp.
    fname = "nature.bmp"
    img = plt.imread(fname)
   
    #3.2 Crie uma função para implementar um colormap definido pelo utilizador.
    cm_red=newCmap([(0,0,0),(1,0,0)], "cm_red", 256)
    cm_green=newCmap([(0,0,0),(0,1,0)], "cm_green", 256)
    cm_blue=newCmap([(0,0,0),(0,0,1)], "cm_blue", 256)
    cm_gray=newCmap([(0,0,0),(1,1,1)], "cm_gray", 256)

    #3.3 Crie uma função que permita visualizar a imagem com um dado colormap.
    showImg(img,fname,"Imagem original: ")

    print("Dimensão Original: " + str(img.shape))  # Imprime as dimensões da imagem original
    
    #3.4 Encoder: Crie uma função para separar a imagem nos seus componentes RGB.
    R, G, B = encoder(img,pad=False,split=True)

    #3.5 Decoder: Crie também a função inversa (que combine os 3 componentes RGB).
    imgRec = decoder(R, G, B, og = None,unpad = False, join= True)
    
    #3.6 Visualize a imagem e cada um dos canais RGB (com o colormap adequado).
    showImg(R,fname,"Img Red: ",cm_red)
    showImg(G,fname,"Img Green: ",cm_green)
    showImg(B,fname,"Img Blue: ",cm_blue)

    #4.1. Encoder: Crie uma função para fazer padding dos canais RGB. 
    '''''
    Obs: Caso a dimensão da imagem não seja múltipla de 32x32, faça padding da mesma, replicando a última linha
    e a última coluna em conformidade.
    '''''
    padded_img, (h, w) = encoder(img, pad=True, split=False)
    print("Dimensão Padded: "+ str(padded_img.shape))  # Imprime as dimensões da imagem padded

    #4.2. Decoder: Crie também a função inversa para remover o padding. 
    '''''
    Obs: Certifique-se de que recupera os canais RGB com a dimensão original, visualizando a imagem original.
    '''''
    unpadded_img = decoder(R,G,B,padded_img = padded_img, og = (h,w),unpad = True,join = False)
    print("Dimensão Unpadded: " + str(unpadded_img.shape))  # Imprime as dimensões da imagem Unpadded
    
    return

if __name__ == "__main__":
    main()