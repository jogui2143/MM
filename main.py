import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import turtle as tt

#1. Compressão de imagens bmp no formato jpeg utilizando um editor de imagem (e.g., GIMP).
#1.1. Comprima as imagens fornecidas segundo o codec JPEG, com qualidade alta (Q=75).
#1.2. Comprima as imagens fornecidas segundo o codec JPEG, com qualidade média (Q=50).
#1.3. Comprima as imagens fornecidas segundo o codec JPEG, com qualidade baixa (Q=25).
#1.4. Compare os resultados e tire conclusões.
#Conclusão: Quanto maior for a taxa de compressão pior será a qualidade da imagem, sendo então o ideal encontrar
#um meio termo que permita ter um certo nível de compressão de imagem e que não comprometa a sua qualidade

#2: criar o encoder e decoder
def encoder(img, pad = False, split = False,RGB_to_YCBCR = False):
  if split:
     R, G, B = splitRGB(img)
     return R,G,B
  elif pad:
    return padding(img)
  elif RGB_to_YCBCR:
     return RGB_to_YCbCr(img)


def decoder(R,G,B,img_ycbcr = None,padded_img = None, og = None, unpad = False,join = False,YCBCR_to_RGB = False):
  if join:
     imgRec = joinRGB(R, G, B)
     return imgRec
  elif unpad:
    return unpadding(padded_img, og)
  elif YCBCR_to_RGB:
     return YCbCr_to_RGB(img_ycbcr)
  

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
'''
Obs: Caso a dimensão da imagem não seja múltipla de 32x32, faça padding da mesma, replicando a última linha
e a última coluna em conformidade.
'''
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


#5
#5.1 Crie uma função para converter a imagem do modelo de cor RGB para o modelo de cor 
#YCbCr. 
def RGB_to_YCbCr(img):
  cm = np.array([[0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312]])

  r = img[:,:,0]
  g = img[:,:,1]
  b = img[:,:,2]

  y = cm[0,0]*r + cm[0,1]*g + cm[0,2]*b
  cb = cm[1,0]*r + cm[1,1]*g + cm[1,2]*b + 128
  cr = cm[2,0]*r + cm[2,1]*g + cm[2,2]*b + 128

  return y, cb, cr

#5.2  Crie também a função inversa (conversão de YCbCr para RGB). Nota: na conversão 
#inversa, garanta que os valores R, G e B obtidos sejam números inteiros no intervalo {0, 1, …, 255}
def YCbCr_to_RGB(img):
  convmatrix = np.array([[0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312]])
  cm_inv = np.linalg.inv(convmatrix)

  y = img[:,:,0]
  cb = img[:,:,1] - 128
  cr = img[:,:,2] - 128

  r = cm_inv[0,0]*y + cm_inv[0,1]*cb + cm_inv[0,2]*cr
  g = cm_inv[1,0]*y + cm_inv[1,1]*cb + cm_inv[1,2]*cr
  b = cm_inv[2,0]*y + cm_inv[2,1]*cb + cm_inv[2,2]*cr

  output_matrix = np.dstack((r, g, b))
  output_matrix = np.round(output_matrix)
  output_matrix[output_matrix > 255] = 255
  output_matrix[output_matrix < 0] = 0
  return output_matrix

def main():
    # 3.1 Leia uma imagem .bmp, e.g., a imagem peppers.bmp.
    fname = "Barns_grand_tetons.bmp"
    img = plt.imread(fname)
    
    #Extrair o pixel [0,0] para verificar se tudo correu bem no final
    original_pixel = img[0, 0]
   
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
    imgRec = decoder(R, G, B, img_ycbcr = None,og = None,unpad = False, join= True,YCBCR_to_RGB = False)
    
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
    unpadded_img = decoder(R,G,B,img_ycbcr = None,padded_img = padded_img, og = (h,w),unpad = True,join = False,YCBCR_to_RGB = False)
    print("Dimensão Unpadded: " + str(unpadded_img.shape))  # Imprime as dimensões da imagem Unpadded

    #5.3
    #5.3.1 Converta os canais RGB para canais YCbCr

    y,cb,cr = encoder(img,pad = False,split = False, RGB_to_YCBCR = True)

    #5.3.2 Visualize cada um dos canais (com o colormap adequado) ----------->>>>>Que color map devemos usar???<<<<<------------
    # Visualizar o canal Y usando mapa de cores em escala de cinza
    showImg(y,fname,'Canal Y (Luminância)','gray')
    
    # Visualizar o canal Cb com mapa de cores apropriado
    showImg(cb,fname,'Canal Cb (Diferença de Azul)','Blues')
    
    # Visualizar o canal Cr com mapa de cores apropriado
    showImg(cr,fname,'Canal Cr (Diferença de Vermelho)','Reds')
    
    #5.4 Decoder: Recupere os canais RGB a partir dos canais YcbCr obtidos. Certifique-se de 
    #que consegue obter os valores originais de RGB (teste, por exemplo, com o pixel de 
    #coordenada [0, 0]).

    #juntar os canais y,cb e cr numa imagem codificada
    encoded_ycbcr_img = np.dstack((y, cb, cr))

    #recuperar a imagem original
    recovered_img = decoder(R,G,B,encoded_ycbcr_img,padded_img = padded_img, og = (h,w),unpad = False,join = False,YCBCR_to_RGB = True)

    # Armazenar os valores RGB do pixel [0,0] da imagem após conversão
    recovered_pixel = recovered_img[0, 0]

    #recuperar os canais RGB 
    R_decoded,G_decoded,B_decoded = splitRGB(recovered_img)

    #verificar se os valores RGB do pixel [0,0] são os mesmos depois de todas as transformações
    print(f'Original RGB pixel [0,0]: {original_pixel}')
    print(f'Recovered RGB pixel [0,0]: {recovered_pixel}')

    return

"""
Ponto de situação:
-->Ver dúvida do colormap
-->Verificar se do ex4 para a frente está tudo ok
-->rushar ex 6 e 7
"""

if __name__ == "__main__":
    main()