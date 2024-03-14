import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from scipy.fftpack import dct, idct
import cv2

def encoder(img = None, pad=False, split=False, RGB_to_YCBCR=False, sub=False, Y=None, Cb=None, Cr=None, subsampling_type=None, interpolation=None,dct = False,dctBlocks=False,step=None,quant = False,fq = None,quant_matrix_Y = None,quant_matrix_CbCr = None):

  if split:
    R, G, B = splitRGB(img)
    return R, G, B

  elif pad:
    return padding(img)

  elif RGB_to_YCBCR:
    return RGB_to_YCbCr(img)

  elif sub:
    return sub_amostragem(Y, Cb, Cr, subsampling_type, interpolation)
  
  elif dct:
     return DCT(Y,Cb,Cr)
  
  elif dctBlocks:
     return DCT_Blocks(Y,Cb,Cr,step)
  
  elif quant:
    Qs_Cro, Qs_Lum = adj_quant_matrix(fq,quant_matrix_Y,quant_matrix_CbCr)
    return quantized_dct(Y, Cb, Cr,Qs_Cro,Qs_Lum,step,fq)
     
 
def decoder(R=None,G=None,B=None,img_ycbcr = None,padded_img = None, og = None, unpad = False,join = False,YCBCR_to_RGB = False, up = False, Y_d = None, Cb_d = None, Cr_d = None, interpolation = None,Invert_DCT = False,invert_dct_Blocks=False,step=None):

  if join:
     imgRec = joinRGB(R, G, B)
     return imgRec

  elif unpad:
    return unpadding(padded_img, og)
  
  elif YCBCR_to_RGB:
     return YCbCr_to_RGB(img_ycbcr)
  
  elif up:
    return upsampling(Y_d,Cb_d,Cr_d,interpolation)
  
  elif Invert_DCT:
     return invertDCT(Y_d,Cb_d,Cr_d)
  
  elif invert_dct_Blocks:
     return invertDCTBlocks(Y_d,Cb_d,Cr_d,step)
  

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

#7. Transformada de Coseno Discreta (DCT).

#7.1.1. Crie uma função para calcular a DCT de um canal completo. Utilize a função scipy.fftpack.dct.
'''
7.1.3. Encoder: Aplique a função desenvolvida em 7.1.1 a Y_d, Cb_d, Cr_d e visualize as
imagens obtidas (Y_dct, Cb_dct, Cr_dct). Sugestão: atendendo à gama ampla de
valores da DCT, visualize as imagens usando uma transformação logarítmica (apta
para compressão de gama), de acordo com o seguinte pseudo-código:
imshow(log(abs(X) + 0.0001))
'''
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

'''
7.1.2. Crie também a função inversa (usando scipy.fftpack.idct).
Nota: para uma matriz, X, com duas dimensões, deverá fazer:
X_dct = dct(dct(X, norm=”ortho”).T, norm=”ortho”).T
'''
'''
7.1.4. Decoder: Aplique a função inversa (7.1.2) e certifique-se de que consegue obter
os valores originais de Y_d, Cb_d e Cr_d. 
'''
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


#7.2. DCT em blocos 8x8

'''
7.2.1. Usando as mesmas funções para cálculo da DCT, crie uma função que calcule a
DCT de um canal completo em blocos BSxBS. 
'''
def DCT_Blocks_Channel(canal, tamanho_bloco=8,Quant = False):
    altura, largura = canal.shape
    canal_dct = np.zeros_like(canal, dtype=float)
    
    for i in range(0, altura, tamanho_bloco):
        for j in range(0, largura, tamanho_bloco):
            bloco = canal[i:i+tamanho_bloco, j:j+tamanho_bloco]
            bloco_padded = np.pad(bloco, ((0, tamanho_bloco - bloco.shape[0]), (0, tamanho_bloco - bloco.shape[1])), 'constant', constant_values=0)
            dct_bloco = dct(dct(bloco_padded.T, norm='ortho').T, norm='ortho')
            
            # Corrigindo a inserção do bloco
            canal_dct[i:i+min(tamanho_bloco, bloco.shape[0]), j:j+min(tamanho_bloco, bloco.shape[1])] = dct_bloco[:bloco.shape[0], :bloco.shape[1]]
    
    return canal_dct

def DCT_Blocks(Y, Cb, Cr, tamanho_bloco=8):  
    Y_dct = DCT_Blocks_Channel(Y, tamanho_bloco)
    Cb_dct = DCT_Blocks_Channel(Cb, tamanho_bloco)
    Cr_dct = DCT_Blocks_Channel(Cr, tamanho_bloco)

    Y_dct_log = np.log(np.abs(Y_dct) + 0.0001)
    Cb_dct_log = np.log(np.abs(Cb_dct) + 0.0001)
    Cr_dct_log = np.log(np.abs(Cr_dct) + 0.0001)

    # Displaying DCT images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_dct_log, cmap='gray')
    plt.title('DCT ' + str(tamanho_bloco) + 'x' + str(tamanho_bloco) + ' of Y')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_dct_log, cmap='gray')
    plt.title('DCT ' + str(tamanho_bloco) + 'x' + str(tamanho_bloco) + ' of Cb')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_dct_log, cmap='gray')
    plt.title('DCT ' + str(tamanho_bloco) + 'x' + str(tamanho_bloco) + ' of Cr')
    plt.tight_layout()
    plt.show()
    
    return Y_dct, Cb_dct, Cr_dct

#7.2.2. Crie também a função inversa (IDCT BSxBS). 
def invertDCTBlocks(Y, Cb, Cr,step):

  # Applying IDCT
  idctOut_Y = np.zeros_like(Y)
  for i in range(0, Y.shape[0], step):
    for j in range(0, Y.shape[1], step):
      dct_bloco = Y[i:i+step, j:j+step]
      idct_bloco = idct(idct(dct_bloco.T, norm='ortho').T, norm='ortho')
      idctOut_Y[i:i+step, j:j+step] = idct_bloco

  idctOut_Cb = np.zeros_like(Cb)
  for i in range(0, Cb.shape[0], step):
    for j in range(0, Cb.shape[1], step):
      dct_bloco = Cb[i:i+step, j:j+step]
      idct_bloco = idct(idct(dct_bloco.T, norm='ortho').T, norm='ortho')
      idctOut_Cb[i:i+step, j:j+step] = idct_bloco

  idctOut_Cr = np.zeros_like(Cr)
  for i in range(0, Cr.shape[0], step):
    for j in range(0, Cr.shape[1], step):
      dct_bloco = Cr[i:i+step, j:j+step]
      idct_bloco = idct(idct(dct_bloco.T, norm='ortho').T, norm='ortho')
      idctOut_Cr[i:i+step, j:j+step] = idct_bloco

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

#8. Quantização.

  #8.1. Crie uma função para quantizar os coeficientes da DCT para cada bloco 8x8

  '''
  8.3. Encoder: Quantize os coeficientes da DCT, usando os seguintes factores de qualidade:
  10, 25, 50, 75 e 100. Visualize as imagens obtidas (Y_q, CB_q e Cr_q).
  '''

def adj_quant_matrix(fator_qualidade,Lum_quant_matrix_std,Cro_quant_matrix_std):
   
  if fator_qualidade >= 50:
     sf = (100 - fator_qualidade) / 50
  
  elif fator_qualidade < 50:
     sf = 50/fator_qualidade

  if sf!=0:
     
     Qs_Lum = [[round(x * sf) for x in sublist] for sublist in Lum_quant_matrix_std]

     Qs_Lum = np.array(Qs_Lum)
     
     Qs_Cro = [[round(x * sf) for x in sublist] for sublist in Cro_quant_matrix_std]

     Qs_Cro = np.array(Qs_Cro)
     

  elif sf == 0:

    Lum_quant_matrix_std = np.array(Lum_quant_matrix_std)
    Cro_quant_matrix_std = np.array(Cro_quant_matrix_std)

    lines_Lum ,cols_Lum = Lum_quant_matrix_std.shape
    lines_Cro ,cols_Cro = Cro_quant_matrix_std.shape

    Qs_Lum = np.ones((lines_Lum,cols_Lum))
    Qs_Cro = np.ones((lines_Cro,cols_Cro))

  lines_Lum ,cols_Lum = Qs_Lum.shape
  lines_Cro ,cols_Cro = Qs_Cro.shape
  
  for i in range(0,lines_Lum):
     for j in range(0,cols_Lum):
        
        if Qs_Lum[i,j] > 255:
           Qs_Lum[i,j] = 255
        
        elif Qs_Lum[i,j] < 1:
           Qs_Lum[i,j] = 1
  
  for i in range(0,lines_Cro):
     for j in range(0,cols_Cro):
        
        if Qs_Cro[i,j] > 255:
           Qs_Cro[i,j] = 255
        
        elif Qs_Cro[i,j] < 1:
           Qs_Cro[i,j] = 1

  return Qs_Cro.astype(np.uint8), Qs_Lum.astype(np.uint8)
     

def quantize_block(dct_block, quant_matrix):
    
    quantized_block = np.round(dct_block / quant_matrix)

    return quantized_block

def quantized_dct(Y_dct, Cb_dct, Cr_dct,quant_matrix_Y,quant_matrix_CbCr,step,fq):
   
  lines_Y, cols_Y = Y_dct.shape
  lines_Cb, cols_Cb = Cb_dct.shape
  lines_Cr, cols_Cr = Cr_dct.shape

  for i in range(0,lines_Y,step):
    for j in range(0,cols_Y,step):
        dct_block = Y_dct[i:i+step, j:j+step]
        dct_block_quant = quantize_block(dct_block,quant_matrix_Y)
        Y_dct[i:i+step, j:j+step] = dct_block_quant

  for i in range(0,lines_Cb,step):
    for j in range(0,cols_Cb,step):
        dct_block = Cb_dct[i:i+step, j:j+step]
        dct_block_quant = quantize_block(dct_block,quant_matrix_CbCr)
        Cb_dct[i:i+step, j:j+step] = dct_block_quant

  for i in range(0,lines_Cr,step):
    for j in range(0,cols_Cr,step):
        dct_block = Cr_dct[i:i+step, j:j+step]
        dct_block_quant = quantize_block(dct_block,quant_matrix_CbCr)
        Cr_dct[i:i+step, j:j+step] = dct_block_quant

  Y_dct_log = np.log(np.abs(Y_dct) + 0.0001)
  Cb_dct_log = np.log(np.abs(Cb_dct) + 0.0001)
  Cr_dct_log = np.log(np.abs(Cr_dct) + 0.0001)

  # Displaying DCT images
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.imshow(Y_dct_log, cmap='gray')
  plt.title('DCT quant '+ str(fq) + ' ' + str(step) + 'x' + str(step) + ' of Y')
  plt.subplot(1, 3, 2)
  plt.imshow(Cb_dct_log, cmap='gray')
  plt.title('DCT quant '+ str(fq) + ' ' + str(step) + 'x' + str(step) + ' of Cb')
  plt.subplot(1, 3, 3)
  plt.imshow(Cr_dct_log, cmap='gray')
  plt.title('DCT quant '+ str(fq) + ' ' + str(step) + 'x' + str(step) + ' of Cr')
  plt.tight_layout()
  plt.show()

  return Y_dct_log, Cb_dct_log, Cr_dct_log

  #8.2. Crie também a função inversa.

  '''
  8.4. Decoder: Desquantize os coeficientes da DCT, usando os mesmos factores de
  qualidade. Visualize as imagens obtidas.
  '''

def dequantize_block(quantized_block, fq,Lum_quant_matrix_std,Cro_quant_matrix_std,Crominancia = False,Luminancia = False): 
  Qs_Cro, Qs_Lum = adj_quant_matrix(fq,Lum_quant_matrix_std,Cro_quant_matrix_std)

  if Crominancia==True:
     dequantized_block = quantized_block * Qs_Cro
  
  elif Luminancia == True:
     dequantized_block = quantized_block * Qs_Lum
     

  
  return dequantized_block

def dequantized_dct(Y_dct_quant, Cb_dct_quant, Cr_dct_quant,quant_matrix_Y,quant_matrix_CbCr,step, fq):
   
   
  
    def apply_idct(Y, Cb, Cr):
          # Applying IDCT
      y_recuperado = idct(idct(Y.T, norm='ortho').T, norm='ortho')
    

      Cb_recuperado = idct(idct(Cb.T, norm='ortho').T, norm='ortho')

      Cr_recuperado = idct(idct(Cr.T, norm='ortho').T, norm='ortho')

        

      return y_recuperado,Cb_recuperado,Cr_recuperado
    
    lines_Y, cols_Y = Y_dct_quant.shape
    lines_Cb, cols_Cb = Cb_dct_quant.shape
    lines_Cr, cols_Cr = Cr_dct_quant.shape

    # Process Y channel
    for i in range(0, lines_Y, step):
        for j in range(0, cols_Y, step):
            dct_block = Y_dct_quant[i:i+step, j:j+step]
            Y_dct_quant[i:i+step, j:j+step] = dequantize_block(dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=False, Luminancia=True)

    # Process Cb channel
    for i in range(0, lines_Cb, step):
        for j in range(0, cols_Cb, step):
            dct_block = Cb_dct_quant[i:i+step, j:j+step]
            Cb_dct_quant[i:i+step, j:j+step] = dequantize_block(dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=True, Luminancia=False)

    # Process Cr channel
    for i in range(0, lines_Cr, step):
        for j in range(0, cols_Cr, step):
            dct_block = Cr_dct_quant[i:i+step, j:j+step]
            Cr_dct_quant[i:i+step, j:j+step] = dequantize_block(dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=True, Luminancia=False)


    y_out, cb_out, cr_out = apply_idct(Y_dct_quant, Cb_dct_quant, Cr_dct_quant)
  
  
    Y_dct_log = np.log(np.abs(y_out) + 0.0001)
    Cb_dct_log = np.log(np.abs(cb_out) + 0.0001)
    Cr_dct_log = np.log(np.abs(cr_out) + 0.0001)


    # Displaying dequantized images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_dct_log, cmap='gray')
    plt.title('DCT dequant '+ str(fq) + ' ' + str(step) + 'x' + str(step) + ' of Y')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_dct_log, cmap='gray')
    plt.title('DCT dequant '+ str(fq) + ' ' + str(step) + 'x' + str(step) + ' of Cb')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_dct_log, cmap='gray')
    plt.title('DCT dequant '+ str(fq) + ' ' + str(step) + 'x' + str(step) + ' of Cr')
    plt.tight_layout()
    plt.show()

    return Y_dct_quant, Cb_dct_quant, Cr_dct_quant


def reconstruct_image(dct_quantized_image, quantization_matrix):
    # Obtendo as dimensões da imagem
    num_rows, num_cols = dct_quantized_image.shape[0], dct_quantized_image.shape[1]
    
    # Criando uma imagem vazia para os resultados da reconstrução
    reconstructed_image = np.zeros((num_rows, num_cols))

    # Processo de reconstrução da imagem bloco por bloco
    for row in range(0, num_rows, 8):
        for col in range(0, num_cols, 8):
            # Passo 1: Desquantizar o bloco
            quantized_block = dct_quantized_image[row:row+8, col:col+8]
            dequantized_block = dequantize_block(quantized_block, quantization_matrix)
            
            # Passo 2: Aplicar a IDCT para reconstruir o bloco
            reconstructed_block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')

            # Passo 3: Armazenar o bloco reconstruído na imagem resultante
            reconstructed_image[row:row+8, col:col+8] = reconstructed_block

    # Assegurar que todos os valores de pixel estejam no intervalo válido [0, 255]
    reconstructed_image[reconstructed_image < 0] = 0
    reconstructed_image[reconstructed_image > 255] = 255

    
      # Displaying DCT images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(reconstructed_image, cmap='gray')
    

    return reconstructed_image.astype(np.uint8)

# Exemplo de uso da função:
# reconstructed_image = reconstruct_image(dct_quantized_image, quantization_matrix)

# Função para desquantizar os canais DCT e aplicar a IDCT
def dequantized_dct_and_reconstruct(Y_dct_quant, Cb_dct_quant, Cr_dct_quant, quant_matrix_Y, quant_matrix_CbCr, step, fq):
    # Função interna para aplicar a IDCT em um canal
    def apply_idct_to_channel(dct_channel):
        channel_idct = np.zeros_like(dct_channel)
        for i in range(0, dct_channel.shape[0], step):
            for j in range(0, dct_channel.shape[1], step):
                channel_idct[i:i+step, j:j+step] = idct(idct(dct_channel[i:i+step, j:j+step].T, norm='ortho').T, norm='ortho')
        return channel_idct
    
    # Desquantização e aplicação da IDCT para cada canal
    Y_idct = apply_idct_to_channel(dequantize_block(Y_dct_quant, quant_matrix_Y, quant_matrix_CbCr, Crominancia=False, Luminancia=True))
    Cb_idct = apply_idct_to_channel(dequantize_block(Cb_dct_quant, quant_matrix_Y, quant_matrix_CbCr, Crominancia=True, Luminancia=False))
    Cr_idct = apply_idct_to_channel(dequantize_block(Cr_dct_quant, quant_matrix_Y, quant_matrix_CbCr, Crominancia=True, Luminancia=False))
    
    # Certificando que os valores de pixels estejam no intervalo correto após IDCT
    Y_idct = np.clip(Y_idct, 0, 255).astype(np.uint8)
    Cb_idct = np.clip(Cb_idct, 0, 255).astype(np.uint8)
    Cr_idct = np.clip(Cr_idct, 0, 255).astype(np.uint8)
    
    return Y_idct, Cb_idct, Cr_idct


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

    '''
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
    '''

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

    print("\n#7\n")

    #7. Transformada de Coseno Discreta (DCT).
    
    #7.1. DCT nos canais completos

    #7.1.1. Crie uma função para calcular a DCT de um canal completo. Utilize a função scipy.fftpack.dct.

    '''
    7.1.3. Encoder: Aplique a função desenvolvida em 7.1.1 a Y_d, Cb_d, Cr_d e visualize as
    imagens obtidas (Y_dct, Cb_dct, Cr_dct). Sugestão: atendendo à gama ampla de
    valores da DCT, visualize as imagens usando uma transformação logarítmica (apta
    para compressão de gama), de acordo com o seguinte pseudo-código:
    imshow(log(abs(X) + 0.0001))
    '''

    Y_d_dct, Cb_d_dct, Cr_d_dct = encoder(None,False,False,False,False,Y_d,Cb_d,Cr_d,None,None,True)

    '''
    7.1.2. Crie também a função inversa (usando scipy.fftpack.idct).
    Nota: para uma matriz, X, com duas dimensões, deverá fazer:
    X_dct = dct(dct(X, norm=”ortho”).T, norm=”ortho”).T
    '''
    #7.1.4. Decoder: Aplique a função inversa (7.1.2) e certifique-se de que consegue obter os valores originais de Y_d, Cb_d e Cr_d. 
    Y_d, Cb_d, Cr_d= decoder(None,None,None,None,None,None,False,False,False,False, Y_d_dct,Cb_d_dct,Cr_d_dct,None,True)

    print("Valores originais Y_d Cb_d Cr_d, pós DCT:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)

    """
    7.2.1. Usando as mesmas funções para cálculo da DCT, crie uma função que calcule a
        DCT de um canal completo em blocos BSxBS.
    7.2.2. Crie também a função inversa (IDCT BSxBS).
    7.2.3. Encoder: Aplique a função desenvolvida (7.2.1) a Y_d, Cb_d, Cr_d com blocos 8x8
        e visualize as imagens obtidas (Y_dct8, Cb_dct8, Cr_dct8).
    7.2.4. Decoder: Aplique a função inversa (7.2.2) e certifique-se de que consegue obter
        os valores originais de Y_d, Cb_d e Cr_d.
    """

    Y_dct8, Cb_dct8, Cr_dct8=encoder(None,False,False,False,False,Y_d,Cb_d,Cr_d,None,None,False,True,8)
    Y_d, Cb_d, Cr_d=decoder(None,None,None,None,None,None,False,False,False,False, Y_dct8, Cb_dct8, Cr_dct8,None,False,True,8)

    print("\nValores originais Y_d Cb_d Cr_d, pós DCT com blocos 8x8:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)

    #7.3. DCT em blocos 64x64.
    Y_dct64, Cb_dct64, Cr_dct64=encoder(None,False,False,False,False,Y_d,Cb_d,Cr_d,None,None,False,True,64)
    Y_d, Cb_d, Cr_d=decoder(None,None,None,None,None,None,False,False,False,False, Y_dct64, Cb_dct64, Cr_dct64,None,False,True,64)

    print("\nValores originais Y_d Cb_d Cr_d, pós DCT com blocos 64x64:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)


    #8. Quantização.
      
      #8.1. Crie uma função para quantizar os coeficientes da DCT para cada bloco 8x8. 
    
    '''
      8.3. Encoder: Quantize os coeficientes da DCT, usando os seguintes factores de qualidade:
      10, 25, 50, 75 e 100. Visualize as imagens obtidas (Y_q, CB_q e Cr_q).
    '''

    matriz_quantizacao_Y = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
    ]

    matriz_quantizacao_CbCr = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
    ]

    Y_dct8_quant, Cb_dct8_quant, Cr_dct8_quant = encoder(None,False,False,False,False,Y_dct8, Cb_dct8, Cr_dct8,None,None,False,False,8,True,50,matriz_quantizacao_Y,matriz_quantizacao_CbCr)

    #Desquantificação teste

    Y_dct_dequant, Cb_dct_dequant, Cr_dct_dequant = dequantized_dct(Y_dct8_quant, Cb_dct8_quant, Cr_dct8_quant,matriz_quantizacao_Y,matriz_quantizacao_CbCr,8, 50)
    #Y_dct_dequant, Cb_dct_dequant, Cr_dct_dequant = dequantized_dct_and_reconstruct(Y_dct8_quant,Cb_dct8_quant , Cr_dct8_quant, matriz_quantizacao_Y, matriz_quantizacao_CbCr, 8, 50)

    #idctOut_Y,idctOut_Cb,idctOut_Cr = invertDCTBlocks(Y_dct_dequant, Cb_dct_dequant, Cr_dct_dequant,8)
   #Y_inv_dct,Cb_inv_dct,Cr_inv_dct = invertDCT(idctOut_Y,idctOut_Cb,idctOut_Cr)

    return

if __name__ == "__main__":
    main()