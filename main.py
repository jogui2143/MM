import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from scipy.fftpack import dct, idct
import cv2



#2: criar o encoder e decoder
def encoder(img, pad=False, split=False, RGB_to_YCBCR=False, sub=False, Y=None, Cb=None, Cr=None, subsampling_type=None, interpolation=None):

  if split:
    R, G, B = splitRGB(img)
    return R, G, B

  elif pad:
    return padding(img)

  elif RGB_to_YCBCR:
    return RGB_to_YCbCr(img)

  elif sub:
    return sub_amostragem(Y, Cb, Cr, subsampling_type, interpolation)
     
 
def decoder(R=None,G=None,B=None,img_ycbcr = None,padded_img = None, og = None, unpad = False,join = False,YCBCR_to_RGB = False, up = False, Y_d = None, Cb_d = None, Cr_d = None, interpolation = None):

  if join:
     imgRec = joinRGB(R, G, B)
     return imgRec

  elif unpad:
    return unpadding(padded_img, og)
  
  elif YCBCR_to_RGB:
     return YCbCr_to_RGB(img_ycbcr)
  
  elif up:
    return upsampling(Y_d,Cb_d,Cr_d,interpolation)
  


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

#6. Sub-amostragem.
'''
  6.1. Crie uma função para sub-amostrar (downsampling) os canais Y, Cb, e Cr, segundo as
  possibilidades definidas pelo codec JPEG, a qual deve devolver Y_d, Cb_d e Cr_d.
  Utilize, para o efeito, a função cv2.resize (biblioteca Computer Vision), testando
  diferentes métodos de interpolação (e.g., linear, cúbica, etc.).
'''

def sub_amostragem(Y, Cb, Cr, subsampling_type, interpolation):
      
  width, height = Y.shape[1], Y.shape[0]

  # os valores aqui defenidos dividir por metade um quarto é a maneira como o JPEG funciona para os varios tipos de subamostragem
  if subsampling_type == '4:2:2':
      
      # reduzimos para metade a resolução horizontal do Cb e do Cr
      Y_d = Y
      Cb_d = cv2.resize(Cb, (width // 2, height), interpolation)
      Cr_d = cv2.resize(Cr, (width // 2, height), interpolation)

  elif subsampling_type == '4:2:0':
      
      # reduzimos para metade a resolução horizontal e vertical do Cb e do Cr
      Y_d = Y
      Cb_d = cv2.resize(Cb, (width // 2, height // 2), interpolation)
      Cr_d = cv2.resize(Cr, (width // 2, height // 2), interpolation)

  return Y_d, Cb_d, Cr_d

#6.2. Crie também a função para efectuar a operação inversa, i.e., upsampling.
def upsampling(Y_d,Cb_d,Cr_d,interpolation):

    Cb_upsampled = cv2.resize(Cb_d, (Y_d.shape[1], Y_d.shape[0]), interpolation)
    Cr_upsampled = cv2.resize(Cr_d, (Y_d.shape[1], Y_d.shape[0]), interpolation)

    return Y_d, Cb_upsampled, Cr_upsampled


def DCT(Y, Cb, Cr):
    # Applying DCT
    Y_dct = dct(dct(Y, norm='ortho').T, norm='ortho').T
    Cb_dct = dct(dct(Cb, norm='ortho').T, norm='ortho').T
    Cr_dct = dct(dct(Cr, norm='ortho').T, norm='ortho').T
    
    # Log transformation for better visualization
    Y_dct_log = np.log(np.abs(Y_dct) + 0.0001)
    Cb_dct_log = np.log(np.abs(Cb_dct) + 0.0001)
    Cr_dct_log = np.log(np.abs(Cr_dct) + 0.0001)

    # Displaying DCT images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_dct_log, cmap='gray')
    plt.title('Log DCT of Y')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_dct_log, cmap='gray')
    plt.title('Log DCT of Cb')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_dct_log, cmap='gray')
    plt.title('Log DCT of Cr')
    plt.tight_layout()
    plt.show()


    return Y_dct,Cb_dct,Cr_dct



def invertDCT(Y_dct, Cb_dct, Cr_dct):  
    # Applying IDCT
    Y_inv_dct = idct(idct(Y_dct, norm='ortho').T, norm='ortho').T
    Cb_inv_dct = idct(idct(Cb_dct, norm='ortho').T, norm='ortho').T
    Cr_inv_dct = idct(idct(Cr_dct, norm='ortho').T, norm='ortho').T

    # Displaying inverse DCT images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_inv_dct, cmap='gray')
    plt.title('Inverse DCT of Y')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_inv_dct, cmap='gray')
    plt.title('Inverse DCT of Cb')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_inv_dct, cmap='gray')
    plt.title('Inverse DCT of Cr')
    plt.tight_layout()
    plt.show()


    return Y_inv_dct,Cb_inv_dct,Cr_inv_dct


def DCTBlocks(Y, Cb, Cr,step):
    # Applying DCT


   
  out_Y = np.zeros(Y.shape)
  for i in range(0, Y.shape[0], step):
    for j in range(0, Y.shape[1], step):
      out_Y[i:i + step, j:j + step] = dct(Y[i:i + step, j:j + step])

  out_Cb = np.zeros(Cb.shape)
  for i in range(0, Cb.shape[0], step):
    for j in range(0, Cb.shape[1], step):
      out_Cb[i:i + step, j:j + step] = dct(Cb[i:i + step, j:j + step])
  
  out_Cr = np.zeros(Cr.shape)
  for i in range(0, Cr.shape[0], step):
    for j in range(0, Cr.shape[1], step):
      out_Cr[i:i + step, j:j + step] = dct(Cr[i:i + step, j:j + step])

  
  
  # Log transformation for better visualization
  Y_dct_log = np.log(np.abs(out_Y) + 0.0001)
  Cb_dct_log = np.log(np.abs(out_Cb) + 0.0001)
  Cr_dct_log = np.log(np.abs(out_Cr) + 0.0001)

  # Displaying DCT images
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.imshow(Y_dct_log, cmap='gray')
  plt.title('Log DCT of Y')
  plt.subplot(1, 3, 2)
  plt.imshow(Cb_dct_log, cmap='gray')
  plt.title('Log DCT of Cb')
  plt.subplot(1, 3, 3)
  plt.imshow(Cr_dct_log, cmap='gray')
  plt.title('Log DCT of Cr')
  plt.tight_layout()
  plt.show()

  return out_Y,out_Cb,out_Cr

def invertDCTBlocks(Y, Cb, Cr,step):
    # Applying IDCT
  idctOut_Y = np.zeros(Y.shape)
  for i in range(0, Y.shape[0], step):
    for j in range(0, Y.shape[1], step):
      idctOut_Y[i:i + step, j:j + step] = idct(Y[i:i + step, j:j + step])

  idctOut_Cb = np.zeros(Cb.shape)
  for i in range(0, Cb.shape[0], step):
    for j in range(0, Cb.shape[1], step):
      idctOut_Cb[i:i + step, j:j + step] = idct(Cb[i:i + step, j:j + step])

  idctOut_Cr = np.zeros(Cr.shape)
  for i in range(0, Cr.shape[0], step):
    for j in range(0, Cr.shape[1], step):
      idctOut_Cr[i:i + step, j:j + step] = idct(Cr[i:i + step, j:j + step])

    # Displaying inverse DCT images
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.imshow(idctOut_Y, cmap='gray')
  plt.title('Inverse DCT of Y')
  plt.subplot(1, 3, 2)
  plt.imshow(idctOut_Cb, cmap='gray')
  plt.title('Inverse DCT of Cb')
  plt.subplot(1, 3, 3)
  plt.imshow(idctOut_Cr, cmap='gray')
  plt.title('Inverse DCT of Cr')
  plt.tight_layout()
  plt.show()


  return idctOut_Y,idctOut_Cb,idctOut_Cr






def main():
    
    # 3.1 Leia uma imagem .bmp, e.g., a imagem peppers.bmp.
    fname = "airport.bmp"
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
    print("\n#4\n")
    print("Dimensão Original: " + str(img.shape))  # Imprime as dimensões da imagem original
    
    #3.4 Encoder: Crie uma função para separar a imagem nos seus componentes RGB.
    R, G, B = encoder(img,pad=False,split=True)

    #3.5 Decoder: Crie também a função inversa (que combine os 3 componentes RGB).
    imgRec = decoder(R, G, B, img_ycbcr = None,og = None,unpad = False, join= True,YCBCR_to_RGB = False)
    
    #3.6 Visualize a imagem e cada um dos canais RGB (com o colormap adequado).
    '''
    showImg(R,fname,"Img Red: ",cm_red)
    showImg(G,fname,"Img Green: ",cm_green)
    showImg(B,fname,"Img Blue: ",cm_blue)
    '''

    #4.1. Encoder: Crie uma função para fazer padding dos canais RGB. 
    '''''
    Obs: Caso a dimensão da imagem não seja múltipla de 32x32, faça padding da mesma, replicando a última linha
    e a última coluna em conformidade.
    '''''
    padded_img, (h, w) = encoder(img, pad=True, split=False)
    print("Dimensão Padded: "+ str(padded_img.shape))  # Imprime as dimensões da imagem padded

    #3.6 com padding
    R,G,B = splitRGB(padded_img)
    showImg(R,fname,"Img Red: ",cm_red)
    showImg(G,fname,"Img Green: ",cm_green)
    showImg(B,fname,"Img Blue: ",cm_blue)

    #4.2. Decoder: Crie também a função inversa para remover o padding. 
    '''''
    Obs: Certifique-se de que recupera os canais RGB com a dimensão original, visualizando a imagem original.
    '''''
    unpadded_img = decoder(R,G,B,img_ycbcr = None,padded_img = padded_img, og = (h,w),unpad = True,join = False,YCBCR_to_RGB = False)
    print("Dimensão Unpadded: " + str(unpadded_img.shape))  # Imprime as dimensões da imagem Unpadded

    #5.3
    #5.3.1 Converta os canais RGB para canais YCbCr (Com paddding)
    y,cb,cr = encoder(padded_img,pad = False,split = False, RGB_to_YCBCR = True)
    
    #5.3.2 Visualize cada um dos canais (com o colormap adequado)
    # Visualizar o canal Y usando mapa de cores em escala de cinza
    showImg(y,fname,'Canal Y (Luminância)','gray')
    
    # Visualizar o canal Cb com mapa de cores apropriado
    showImg(cb,fname,'Canal Cb (Diferença de Azul)','gray')
    
    # Visualizar o canal Cr com mapa de cores apropriado
    showImg(cr,fname,'Canal Cr (Diferença de Vermelho)','gray')
    
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
    print("\n#5\n")
    print(f'Original RGB pixel [0,0]: {original_pixel}')
    print(f'Recovered RGB pixel [0,0]: {recovered_pixel}')

    '''
    6.1. Crie uma função para sub-amostrar (downsampling) os canais Y, Cb, e Cr, segundo as
    possibilidades definidas pelo codec JPEG, a qual deve devolver Y_d, Cb_d e Cr_d.
    Utilize, para o efeito, a função cv2.resize (biblioteca Computer Vision), testando
    diferentes métodos de interpolação (e.g., linear, cúbica, etc.).
    
    6.3. Encoder: Obtenha e visualize os canais Y_d, Cb_d e Cr_d com downsampling 4:2:0.
    Apresente as dimensões das matrizes correspondentes.

    6.4. Decoder: Reconstrua e visualize os canais Y, Cb e Cr. Compare-os com os originais.
    '''

    print("\n#6\n")

    # 4:2:0 & LINEAR
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False,True, y ,cb ,cr, "4:2:0",cv2.INTER_LINEAR)
  
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_d, cmap='gray')
    plt.title('Y downsampling 4:2:0 (LINEAR)')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_d, cmap='gray')
    plt.title(f'Cb downsampling 4:2:0 (LINEAR)')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_d, cmap='gray')
    plt.title(f'Cr downsampling 4:2:0 (LINEAR)')
    plt.tight_layout()
    plt.show()

    print("---[downsampling 4:2:0 (LINEAR)]---\n")
    print("Dimensões de Y_d:", Y_d.shape)
    print("Dimensões de Cb_d:", Cb_d.shape)
    print("Dimensões de Cr_d:", Cr_d.shape)

    Y, Cb, Cr = decoder(None,None,None,None,None,None,False, False, False, True, Y_d , Cb_d , Cr_d, cv2.INTER_LINEAR)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title('Y upsampling  4:2:0 (LINEAR)')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb, cmap='gray')
    plt.title(f'Cb upsampling  4:2:0 (LINEAR)')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr, cmap='gray')
    plt.title(f'Cr upsampling  4:2:0 (LINEAR)')
    plt.tight_layout()
    plt.show()

    print("\n---[upsampling 4:2:0 (LINEAR)]---\n")
    print("Dimensões de Y:", Y.shape)
    print("Dimensões de Cb:", Cb.shape)
    print("Dimensões de Cr:", Cr.shape)

    # 4:2:0 & CUBIC
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False, True, y ,cb ,cr, "4:2:0",cv2.INTER_CUBIC)
   
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_d, cmap='gray')
    plt.title('Y downsampling 4:2:0 (CUBIC)')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_d, cmap='gray')
    plt.title(f'Cb downsampling 4:2:0 (CUBIC)')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_d, cmap='gray')
    plt.title(f'Cr downsampling 4:2:0 (CUBIC)')
    plt.tight_layout()
    plt.show()

    print("\n---[downsampling 4:2:0 (CUBIC)]---\n")
    print("Dimensões de Y_d:", Y_d.shape)
    print("Dimensões de Cb_d:", Cb_d.shape)
    print("Dimensões de Cr_d:", Cr_d.shape)

    Y, Cb, Cr = decoder(None,None,None,None,None,None,False, False, False, True, Y_d , Cb_d , Cr_d, cv2.INTER_CUBIC)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title('Y upsampling  4:2:0 (CUBIC)')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb, cmap='gray')
    plt.title(f'Cb upsampling  4:2:0 (CUBIC)')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr, cmap='gray')
    plt.title(f'Cr upsampling  4:2:0 (CUBIC)')
    plt.tight_layout()
    plt.show()
    
    print("\n---[upsampling 4:2:0 (CUBIC)]---\n")
    print("Dimensões de Y:", Y.shape)
    print("Dimensões de Cb:", Cb.shape)
    print("Dimensões de Cr:", Cr.shape)
    
    # 4:2:2 & LINEAR
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False, True, y ,cb ,cr, "4:2:2",cv2.INTER_LINEAR)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_d, cmap='gray')
    plt.title('Y downsampling 4:2:2 (LINEAR)')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_d, cmap='gray')
    plt.title(f'Cb downsampling 4:2:2 (LINEAR)')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_d, cmap='gray')
    plt.title(f'Cr downsampling 4:2:2 (LINEAR)')
    plt.tight_layout()
    plt.show()

    print("\n---[downsampling 4:2:2 (LINEAR)]---\n")
    print("Dimensões de Y_d:", Y_d.shape)
    print("Dimensões de Cb_d:", Cb_d.shape)
    print("Dimensões de Cr_d:", Cr_d.shape)

    Y, Cb, Cr = decoder(None,None,None,None,None,None,False, False, False, True, Y_d , Cb_d , Cr_d, cv2.INTER_LINEAR)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title('Y upsampling  4:2:2 (LINEAR)')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb, cmap='gray')
    plt.title(f'Cb upsampling  4:2:2 (LINEAR)')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr, cmap='gray')
    plt.title(f'Cr upsampling  4:2:2 (LINEAR)')
    plt.tight_layout()
    plt.show()
    
    print("\n---[upsampling 4:2:2 (LINEAR)]---\n")
    print("Dimensões de Y:", Y.shape)
    print("Dimensões de Cb:", Cb.shape)
    print("Dimensões de Cr:", Cr.shape)

    # 4:2:2 & CUBIC
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False, True, y ,cb ,cr, "4:2:2",cv2.INTER_CUBIC)
   
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_d, cmap='gray')
    plt.title('Y downsampling 4:2:2 (CUBIC)')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_d, cmap='gray')
    plt.title(f'Cb downsampling 4:2:2 (CUBIC)')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_d, cmap='gray')
    plt.title(f'Cr downsampling 4:2:2 (CUBIC)')
    plt.tight_layout()
    plt.show()

    print("\n---[downsampling 4:2:2 (CUBIC)]---\n")
    print("Dimensões de Y_d:", Y_d.shape)
    print("Dimensões de Cb_d:", Cb_d.shape)
    print("Dimensões de Cr_d:", Cr_d.shape)

    Y, Cb, Cr = decoder(None,None,None,None,None,None,False, False, False, True, Y_d , Cb_d , Cr_d, cv2.INTER_CUBIC)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title('Y upsampling  4:2:2 (CUBIC)')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb, cmap='gray')
    plt.title(f'Cb upsampling  4:2:2 (CUBIC)')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr, cmap='gray')
    plt.title(f'Cr upsampling  4:2:2 (CUBIC)')
    plt.tight_layout()
    plt.show()
    
    print("\n---[upsampling 4:2:2 (CUBIC)]---\n")
    print("Dimensões de Y:", Y.shape)
    print("Dimensões de Cb:", Cb.shape)
    print("Dimensões de Cr:", Cr.shape)
  

    y,cb,cr=DCT(y,cb,cr)
    y,cb,cr=invertDCT(y,cb,cr)
    y,cb,cr=DCTBlocks(y, cb, cr,8)
    y,cb,cr=invertDCTBlocks(y, cb, cr,8)
    y,cb,cr=DCTBlocks(y, cb, cr,64)
    y,cb,cr=invertDCTBlocks(y, cb, cr,64)
    

    return


"""
Ponto de situação:
-->rushar ex  7
-->Perguntar ao professor as diferenças entre a interpolação e se os resultados visuais fazem sentido (Nós não notamos bem)
    
"""

if __name__ == "__main__":
    main()