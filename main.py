import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from scipy.fftpack import dct, idct
import cv2

def encoder(img = None, pad=False, split=False, RGB_to_YCBCR=False, sub=False, Y=None, Cb=None, Cr=None, subsampling_type=None, interpolation=None,dct = False,dctBlocks=False,step=None,quant = False,fq = None,quant_matrix_Y = None,quant_matrix_CbCr = None,dpcmfds=False,channel=None):

  if split:
    R, G, B = splitRGB(img)
    return R, G, B

  elif pad:
    return padding(img)

  elif RGB_to_YCBCR:
    R,G,B = splitRGB(img)
    return rgb_to_ycbcr(R,G,B)

  elif sub:
    return sub_amostragem(Y, Cb, Cr, subsampling_type, interpolation)
  
  elif dct:
     return DCT(Y,Cb,Cr)
  
  elif dctBlocks:
     return DCT_Blocks(Y,Cb,Cr,step)
  
  elif quant:
    Qs_Cro, Qs_Lum = adj_quant_matrix(fq,quant_matrix_Y,quant_matrix_CbCr)
    return quantized_dct(Y, Cb, Cr,Qs_Lum,Qs_Cro,step,fq)
  
  elif dpcmfds:
     return DPCM(Y),DPCM(Cb),DPCM(Cr)
     
 
def decoder(R=None,G=None,B=None,img = None,padded_img = None, og = None, unpad = False,join = False,YCBCR_to_RGB = False, up = False, Y = None, Cb = None, Cr = None, interpolation = None,Invert_DCT = False,invert_dct_Blocks=False,step=None,dequant = False,quant_matrix_Y = None,quant_matrix_CbCr = None, fq = None,DPCM = False, dpcm = None, reconstr = False):

  if join:
     imgRec = joinRGB(R, G, B)
     return imgRec

  elif unpad:
    return unpadding(padded_img, og)
  
  elif YCBCR_to_RGB:
     return ycbcr_to_rgb(Y,Cb,Cr)
  
  elif up:
    return upsampling(Y,Cb,Cr,interpolation)
  
  elif Invert_DCT:
     return invertDCT(Y,Cb,Cr)
  
  elif invert_dct_Blocks:
     return invertDCTBlocks(Y,Cb,Cr,step)
  
  elif dequant:        
     return dequantized_dct(Y,Cb,Cr,quant_matrix_Y,quant_matrix_CbCr,step,fq)

  elif DPCM:
     return invDPCM(dpcm[0]),invDPCM(dpcm[1]),invDPCM(dpcm[2])
  
  elif reconstr:
     Y_inv_dpcm = invDPCM(Y)
     Cb_inv_dpcm = invDPCM(Cb)
     Cr_inv_dpcm = invDPCM(Cr)
     Y_dct_deq, Cb_dct_deq, Cr_dct_deq = dequantized_dct(Y_inv_dpcm,Cb_inv_dpcm,Cr_inv_dpcm,quant_matrix_Y,quant_matrix_CbCr,step,fq)
     Y_idct_invblock, Cb_idct_invblock, Cr_idct_invblock = invertDCTBlocks(Y_dct_deq, Cb_dct_deq, Cr_dct_deq,step)
     Y_up,Cb_up,Cr_up = upsampling(Y_idct_invblock, Cb_idct_invblock, Cr_idct_invblock,interpolation)
     R,G,B = ycbcr_to_rgb(Y_up,Cb_up,Cr_up)
     img_p = joinRGB(R,G,B)
     return unpadding(img_p,og)
  

#3.2 Crie uma função para implementar um colormap definido pelo utilizador.
def newCmap(keyColors = [(0,0,0),(1,1,1)], name = "gray", N= 256):
    cm = clr.LinearSegmentedColormap.from_list(name, keyColors, N)
    return cm

#3.3 Crie uma função que permita visualizar a imagem com um dado colormap.
def showImg(img, fname="", caption="", cmap=None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(caption + fname)
    plt.show()  # Exibe a imagem

#3.4. Encoder: Crie uma função para separar a imagem nos seus componentes RGB.
def splitRGB(img):
    R = img[:, :, 0] 
    G = img[:, :, 1] 
    B = img[:, :, 2]
    return R, G, B

#3.5. Decoder: Crie também a função inversa (que combine os 3 componentes RGB).
def joinRGB(R,G,B):
    nl, nc = R.shape
    imgRec = np.zeros((nl, nc, 3), dtype=np.uint8)
    imgRec[:, :, 0] = R
    imgRec[:, :, 1] = G
    imgRec[:, :, 2] = B
    return imgRec 

'''''
#4.1. Encoder: Crie uma função para fazer padding dos canais RGB. 
Obs: Caso a dimensão da imagem não seja múltipla de 32x32, faça padding da mesma, replicando a última linha
e a última coluna em conformidade.
'''''

def padding(img):
  
  h,w = img.shape[:2]
  p1, p2, p3 = splitRGB(img)

  r,c = p1.shape

  v_pad = 32 - (r % 32) if r % 32 > 0 else 0
  h_pad = 32 - (c % 32) if c % 32 > 0 else 0  

  p1 = np.vstack([p1, np.repeat(np.array(p1[-1,:], ndmin = 2), v_pad, axis=0)])
  p1 = np.hstack([p1, np.repeat(np.array(p1[:,-1], ndmin = 2), h_pad, axis=0).T])

  p2 = np.vstack([p2, np.repeat(np.array(p2[-1,:], ndmin = 2), v_pad, axis=0)])
  p2 = np.hstack([p2, np.repeat(np.array(p2[:,-1], ndmin = 2), h_pad, axis=0).T])

  p3 = np.vstack([p3, np.repeat(np.array(p3[-1,:], ndmin = 2), v_pad, axis=0)])
  p3 = np.hstack([p3, np.repeat(np.array(p3[:,-1], ndmin = 2), h_pad, axis=0).T])

  return np.dstack((p1, p2, p3)), (h,w)

'''''
#4.2. Decoder: Crie também a função inversa para remover o padding. 
Obs: Certifique-se de que recupera os canais RGB com a dimensão original, visualizando a imagem original.
'''''

def unpadding(img, og):
  return img[:og[0], :og[1], :]


#5.1 Crie uma função para converter a imagem do modelo de cor RGB para o modelo de cor YCbCr. 

def rgb_to_ycbcr(R,G,B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B #Luminancia  
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128 # Crominancia Azul
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128 # Crominancia Vermelho
    return Y,Cb,Cr

'''
5.2  Crie também a função inversa (conversão de YCbCr para RGB). Nota: na conversão 
inversa, garanta que os valores R, G e B obtidos sejam números inteiros no intervalo {0, 1, …, 255}
'''

def ycbcr_to_rgb(Y, Cb, Cr,show = False):        

    Tc = np.array([[0.299, 0.587, 0.114],
              [-0.168736, -0.331264, 0.5],
              [0.5, -0.418688, -0.081312]])
    
    Tci = np.linalg.inv(Tc)
   
    R = Tci[0,0]*Y + Tci[0,1]*(Cb - 128) + Tci[0,2]*(Cr - 128)
    R[R>255] = 255
    R[R<0] = 0
    R = np.round(R).astype(np.uint8)
    G  = Tci[1,0]*Y + Tci[1,1]*(Cb - 128) + Tci[1,2]*(Cr - 128)
    #print(G[:8, :8])
    G[G>255] = 255
    G[G<0] = 0
    G = np.round(G).astype(np.uint8)
    B = Tci[2,0]*Y + Tci[2,1]*(Cb - 128) + Tci[2,2]*(Cr - 128)
    B[B>255] = 255
    B[B<0] = 0
    B = np.round(B).astype(np.uint8)

    return R,G,B

'''
6.1. Crie uma função para sub-amostrar (downsampling) os canais Y, Cb, e Cr, segundo as
possibilidades definidas pelo codec JPEG, a qual deve devolver Y_d, Cb_d e Cr_d.
Utilize, para o efeito, a função cv2.resize (biblioteca Computer Vision), testando
diferentes métodos de interpolação (e.g., linear, cúbica, etc.).
'''

def sub_amostragem(Y, Cb, Cr, subsampling_type, interpolation):
      
  width, height = Y.shape[1], Y.shape[0]

  if subsampling_type == '4:2:2':
      
      Y_d = Y
      Cb_d = cv2.resize(Cb, (width // 2, height), interpolation)
      Cr_d = cv2.resize(Cr, (width // 2, height), interpolation)

  elif subsampling_type == '4:2:0':
      
      Y_d = Y
      Cb_d = cv2.resize(Cb, (width // 2, height // 2), interpolation)
      Cr_d = cv2.resize(Cr, (width // 2, height // 2), interpolation)

  return Y_d, Cb_d, Cr_d

#6.2. Crie também a função para efectuar a operação inversa, i.e., upsampling.
def upsampling(Y_d,Cb_d,Cr_d,interpolation):

    Cb_upsampled = cv2.resize(Cb_d, (Y_d.shape[1], Y_d.shape[0]), interpolation)
    Cr_upsampled = cv2.resize(Cr_d, (Y_d.shape[1], Y_d.shape[0]), interpolation)

    return Y_d, Cb_upsampled, Cr_upsampled

'''
7. Transformada de Coseno Discreta (DCT).
7.1.1. Crie uma função para calcular a DCT de um canal completo. Utilize a função scipy.fftpack.dct.
7.1.3. Encoder: Aplique a função desenvolvida em 7.1.1 a Y_d, Cb_d, Cr_d e visualize as
imagens obtidas (Y_dct, Cb_dct, Cr_dct). Sugestão: atendendo à gama ampla de
valores da DCT, visualize as imagens usando uma transformação logarítmica (apta
para compressão de gama), de acordo com o seguinte pseudo-código:
imshow(log(abs(X) + 0.0001))
'''

def DCT(Y, Cb, Cr):
    
    Y_dct = dct(dct(Y, norm='ortho').T, norm='ortho').T
    Cb_dct = dct(dct(Cb, norm='ortho').T, norm='ortho').T
    Cr_dct = dct(dct(Cr, norm='ortho').T, norm='ortho').T
    
    Y_dct_log = np.log(np.abs(Y_dct) + 0.0001)
    Cb_dct_log = np.log(np.abs(Cb_dct) + 0.0001)
    Cr_dct_log = np.log(np.abs(Cr_dct) + 0.0001)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_dct_log, cmap='gray')
    plt.title('DCT of Y')
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_dct_log, cmap='gray')
    plt.title('DCT of Cb')
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_dct_log, cmap='gray')
    plt.title('DCT of Cr')
    plt.tight_layout()
    plt.show()

    return Y_dct,Cb_dct,Cr_dct

'''
7.1.2. Crie também a função inversa (usando scipy.fftpack.idct).
Nota: para uma matriz, X, com duas dimensões, deverá fazer:
X_dct = dct(dct(X, norm=”ortho”).T, norm=”ortho”).T
7.1.4. Decoder: Aplique a função inversa (7.1.2) e certifique-se de que consegue obter
os valores originais de Y_d, Cb_d e Cr_d. 
'''

def invertDCT(Y_dct, Cb_dct, Cr_dct):
    
    Y_inv_dct = idct(idct(Y_dct, norm='ortho').T, norm='ortho').T
    Cb_inv_dct = idct(idct(Cb_dct, norm='ortho').T, norm='ortho').T
    Cr_inv_dct = idct(idct(Cr_dct, norm='ortho').T, norm='ortho').T

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

def single_DCT_log(channel):
   return np.log(np.abs(channel) + .0001)

def mult_DCT_log(channels):
  return single_DCT_log(channels[0]),single_DCT_log(channels[1]),single_DCT_log(channels[2])

def display_images(images, titles, colormap='gray'):
    
    plt.figure(figsize=(12, 4))
    num_images = len(images)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap=colormap)
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()

'''
7.2. DCT em blocos 8x8
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
            
            canal_dct[i:i+min(tamanho_bloco, bloco.shape[0]), j:j+min(tamanho_bloco, bloco.shape[1])] = dct_bloco[:bloco.shape[0], :bloco.shape[1]]
    
    return canal_dct

def DCT_Blocks(Y, Cb, Cr, tamanho_bloco=8):  
    Y_dct = DCT_Blocks_Channel(Y, tamanho_bloco)
    Cb_dct = DCT_Blocks_Channel(Cb, tamanho_bloco)
    Cr_dct = DCT_Blocks_Channel(Cr, tamanho_bloco)

    Y_dct_log = np.log(np.abs(Y_dct) + 0.0001)
    Cb_dct_log = np.log(np.abs(Cb_dct) + 0.0001)
    Cr_dct_log = np.log(np.abs(Cr_dct) + 0.0001)

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

'''
8.1. Crie uma função para quantizar os coeficientes da DCT para cada bloco 8x8
8.3. Encoder: Quantize os coeficientes da DCT, usando os seguintes factores de qualidade:
10, 25, 50, 75 e 100. Visualize as imagens obtidas (Y_q, CB_q e Cr_q).
'''

def adj_quant_matrix(fator_qualidade,Lum_quant_matrix_std,Cro_quant_matrix_std):
  if fator_qualidade < 50:
     fator = 50 / fator_qualidade
  
  else:
     fator = (100-fator_qualidade) / 50

  if fator_qualidade == 0:
      return np.ones_like(Cro_quant_matrix_std),np.ones_like(Lum_quant_matrix_std)

  Qs_Lum = np.round((Lum_quant_matrix_std * fator))
  Qs_Cro = np.round((Cro_quant_matrix_std * fator))

  Qs_Cro[Qs_Cro < 1] = 1
  Qs_Cro[Qs_Cro > 255] = 255

  Qs_Lum[Qs_Lum < 1] = 1
  Qs_Lum[Qs_Lum > 255] = 255

  return Qs_Cro.astype(np.uint8),Qs_Lum.astype(np.uint8)


def quantized_dct(Y_dct, Cb_dct, Cr_dct,quant_matrix_Y,quant_matrix_CbCr,step,fq):
   
  lines_Y, cols_Y = Y_dct.shape
  lines_Cb, cols_Cb = Cb_dct.shape
  lines_Cr, cols_Cr = Cr_dct.shape

  canal_quantizado_Y = np.zeros_like(Y_dct)
  canal_quantizado_Cb = np.zeros_like(Cb_dct)
  canal_quantizado_Cr = np.zeros_like(Cr_dct)

  for i in range(0,lines_Y,step):
    for j in range(0,cols_Y,step):
        dct_block = Y_dct[i:i+step, j:j+step]
        dct_block_quant = np.round(dct_block / quant_matrix_Y)
        canal_quantizado_Y[i:i+step, j:j+step] = dct_block_quant

  for i in range(0,lines_Cb,step):
    for j in range(0,cols_Cb,step):
        dct_block = Cb_dct[i:i+step, j:j+step]
        dct_block_quant = np.round(dct_block / quant_matrix_CbCr)
        canal_quantizado_Cb[i:i+step, j:j+step] = dct_block_quant

  for i in range(0,lines_Cr,step):
    for j in range(0,cols_Cr,step):
        dct_block = Cr_dct[i:i+step, j:j+step]
        dct_block_quant = np.round(dct_block / quant_matrix_CbCr)
        canal_quantizado_Cr[i:i+step, j:j+step] = dct_block_quant

  Y_dct_log = np.log(np.abs(canal_quantizado_Y) + 0.0001)
  Cb_dct_log = np.log(np.abs(canal_quantizado_Cb) + 0.0001)
  Cr_dct_log = np.log(np.abs(canal_quantizado_Cr) + 0.0001)

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

  return canal_quantizado_Y.astype(np.int16), canal_quantizado_Cb.astype(np.int16), canal_quantizado_Cr.astype(np.int16)

'''
8.2. Crie também a função inversa.
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

    lines_Y, cols_Y = Y_dct_quant.shape
    lines_Cb, cols_Cb = Cb_dct_quant.shape
    lines_Cr, cols_Cr = Cr_dct_quant.shape

    canal_Y_dct_desquantizado = np.zeros_like(Y_dct_quant)

    canal_Cb_dct_desquantizado = np.zeros_like(Cb_dct_quant)

    canal_Cr_dct_desquantizado = np.zeros_like(Cr_dct_quant)

    #Y
    for i in range(0, lines_Y, step):
        for j in range(0, cols_Y, step):
            quant_dct_block = Y_dct_quant[i:i+step, j:j+step]
            canal_Y_dct_desquantizado[i:i+step, j:j+step] = dequantize_block(quant_dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=False, Luminancia=True)

    #Cb
    for i in range(0, lines_Cb, step):
        for j in range(0, cols_Cb, step):
            quant_dct_block = Cb_dct_quant[i:i+step, j:j+step]
            canal_Cb_dct_desquantizado[i:i+step, j:j+step] = dequantize_block(quant_dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=True, Luminancia=False)

    #Cr
    for i in range(0, lines_Cr, step):
        for j in range(0, cols_Cr, step):
            quant_dct_block = Cr_dct_quant[i:i+step, j:j+step]
            canal_Cr_dct_desquantizado[i:i+step, j:j+step] = dequantize_block(quant_dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=True, Luminancia=False)

    Y_dct_log = np.log(np.abs(canal_Y_dct_desquantizado) + 0.0001)
    Cb_dct_log = np.log(np.abs(canal_Cb_dct_desquantizado) + 0.0001)
    Cr_dct_log = np.log(np.abs(canal_Cr_dct_desquantizado) + 0.0001)

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

    return canal_Y_dct_desquantizado, canal_Cb_dct_desquantizado, canal_Cr_dct_desquantizado

'''
9. Codificação DPCM dos coeficientes DC.
9.1. Crie uma função para realizar a codificação dos coeficientes DC de cada bloco. Em cada
bloco, substitua o valor DC pelo valor da diferença. 
'''

def DPCM(quantizada):
    
    altura, largura = quantizada.shape
    imagem_codificada_dpcm = np.zeros_like(quantizada)
    bloco_anterior_dc = 0

    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            bloco_quantizado = quantizada[i:i+8, j:j+8]
            dc_atual = bloco_quantizado[0, 0]
            diferenca = dc_atual - bloco_anterior_dc
            bloco_anterior_dc = dc_atual

            bloco_codificado_dpcm = bloco_quantizado.copy()
            bloco_codificado_dpcm[0, 0] = diferenca
            imagem_codificada_dpcm[i:i+8, j:j+8] = bloco_codificado_dpcm

    return imagem_codificada_dpcm

def invDPCM(imagem_codificada_dpcm):
    
    altura, largura = imagem_codificada_dpcm.shape
    imagem_descodificada = np.zeros_like(imagem_codificada_dpcm)
    bloco_anterior_dc = 0

    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            bloco_codificado_dpcm = imagem_codificada_dpcm[i:i+8, j:j+8]
            diferenca = bloco_codificado_dpcm[0, 0]
            bloco_descodificado = bloco_codificado_dpcm.copy()
            bloco_descodificado[0, 0] = bloco_anterior_dc + diferenca
            imagem_descodificada[i:i+8, j:j+8] = bloco_descodificado
            bloco_anterior_dc = bloco_descodificado[0, 0]

    return imagem_descodificada

'''
10.3. Crie uma função para cálculo da imagem das diferenças (entre o canal Y da original e
da descompactada).
'''

def channel_diference(Y,img_reconstr):
    
    padded_img_r, (h, w) = padding(img_reconstr)

    R,G,B = splitRGB(padded_img_r)
    
    Y_r, Cb_r, Cr_r = rgb_to_ycbcr(R,G,B)
    
    diff = np.abs(Y-Y_r)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_reconstr, None)
    plt.title('Img Reconstr')
    
    plt.subplot(1, 2, 2)
    plt.imshow(diff, cmap='gray')
    plt.title('Imagem diferenças')
    
    plt.tight_layout()
    plt.show()

'''
10.4. Crie uma função para calculo das métricas de distorção MSE, RMSE, SNR, PSNR,
max_diff e avg_diff (por comparação da imagem original com a descompactada).
'''

def distorcao(im_original, im_recuperada,Y,Yr):
    
    im_original = im_original.astype(float)
    im_recuperada = im_recuperada.astype(float)
    
    diferenca = np.abs(im_original - im_recuperada)
    diffY= np.abs(Y-Yr)
    
    n_linhas = im_original.shape[0]
    n_colunas = im_original.shape[1]

    mse = np.sum((diferenca)**2) / (n_linhas * n_colunas)

    rmse = np.sqrt(mse)
    
    p = np.sum((im_original)**2)/(n_linhas * n_colunas)
    
    snr = 10 * np.log10(p / mse)
    
    psnr = 10 * np.log10((np.max(im_original)**2) / mse)
    
    max_diff = np.max(diffY)
    
    avg_diff = np.mean(diffY)

    print("\nMSE = "+ str(mse) + "\nRMSE = " + str(rmse) + "\nSNR = " + str(snr) + "\nPSNR = " + str(psnr) + "\n\nMax diff: " + str(max_diff) + "\nAvg diff: " + str(avg_diff))


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

    #3.4. Encoder: Crie uma função para separar a imagem nos seus componentes RGB
    R,G,B = splitRGB(img)
    showImg(R,fname,"Img Red: ",cm_red)
    showImg(G,fname,"Img Green: ",cm_green)
    showImg(B,fname,"Img Blue: ",cm_blue)

    #3.5 Decoder: Crie também a função inversa (que combine os 3 componentes RGB).
    #imgRec = decoder(R, G, B, img_ycbcr = None,og = None,unpad = False, join= True,YCBCR_to_RGB = False)

    '''''
    4.1. Encoder: Crie uma função para fazer padding dos canais RGB. 
    Obs: Caso a dimensão da imagem não seja múltipla de 32x32, faça padding da mesma, replicando a última linha
    e a última coluna em conformidade.
    '''''

    print("\n#4\n")

    print("Dimensão Original: " + str(img.shape))

    padded_img, (h, w) = encoder(img, pad=True, split=False)
    print("Dimensão Padded: "+ str(padded_img.shape))

    #3.6 com padding
    R,G,B = splitRGB(padded_img)
    showImg(R,fname,"Img Red Padded: ",cm_red)
    showImg(G,fname,"Img Green Padded: ",cm_green)
    showImg(B,fname,"Img Blue Padded: ",cm_blue)

    '''
    4.2. Decoder: Crie também a função inversa para remover o padding. 
    Obs: Certifique-se de que recupera os canais RGB com a dimensão original, visualizando a imagem original.
    '''
    unpadded_img = decoder(R,G,B,img = None,padded_img = padded_img, og = (h,w),unpad = True,join = False,YCBCR_to_RGB = False)
    print("Dimensão Unpadded: " + str(unpadded_img.shape))  # Imprime as dimensões da imagem Unpadded

    #5.3.1 Converta os canais RGB para canais YCbCr
    y,cb,cr = encoder(padded_img,pad = False,split = False, RGB_to_YCBCR = True)
    
    #copia do canal Y original para o ex 10
    Y_og_ex10 = y.copy()
    
    #5.3.2 Visualize cada um dos canais (com o colormap adequado)
    showImg(y,fname,'Canal Y ',cm_gray)
    showImg(cb,fname,'Canal Cb ' ,cm_gray)
    showImg(cr,fname,'Canal Cr ',cm_gray)
    
    '''
    5.4 Decoder: Recupere os canais RGB a partir dos canais YcbCr obtidos. Certifique-se de 
    que consegue obter os valores originais de RGB (teste, por exemplo, com o pixel de 
    coordenada [0, 0]).
    '''

    #recuperar a imagem original
    R_recovered,G_recovered,B_recovered = decoder(R,G,B,None,padded_img,(h,w),False,False,True,False,y,cb,cr,None,False,False,None,False,None,None,None,False,None,False)

    recovered_img = joinRGB(R_recovered,G_recovered,B_recovered)

    #armazenar os valores RGB do pixel [0,0] da imagem após conversão
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
    '''

    dimensao = "4:2:2"
    interpolacao = cv2.INTER_LINEAR

    print("\n#6\n")

    # 4:2:2 & LINEAR
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False, True, y ,cb ,cr,dimensao,interpolacao)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y_d, cm_gray)
    plt.title('Y downsampling ' + dimensao)
    plt.subplot(1, 3, 2)
    plt.imshow(Cb_d, cm_gray)
    plt.title(f'Cb downsampling ' + dimensao)
    plt.subplot(1, 3, 3)
    plt.imshow(Cr_d, cm_gray)
    plt.title(f'Cr downsampling ' + dimensao)
    plt.tight_layout()
    plt.show()

    print('---[downsampling ' + dimensao + ']---\n')
    print("Dimensões de Y_d:", Y_d.shape)
    print("Dimensões de Cb_d:", Cb_d.shape)
    print("Dimensões de Cr_d:", Cr_d.shape)
    
    '''
    6.4. Decoder: Reconstrua e visualize os canais Y, Cb e Cr. Compare-os com os originais.
    '''

    Y, Cb, Cr = decoder(None,None,None,None,None,None,False, False, False, True, Y_d , Cb_d , Cr_d, cv2.INTER_LINEAR)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title('Y upsampling ' + dimensao)
    plt.subplot(1, 3, 2)
    plt.imshow(Cb, cmap='gray')
    plt.title(f'Cb upsampling ' + dimensao)
    plt.subplot(1, 3, 3)
    plt.imshow(Cr, cmap='gray')
    plt.title(f'Cr upsampling ' + dimensao)
    plt.tight_layout()
    plt.show()
    
    print('\n---[upsampling ' + dimensao + ']---\n')
    print("Dimensões de Y:", Y.shape)
    print("Dimensões de Cb:", Cb.shape)
    print("Dimensões de Cr:", Cr.shape)

    print("\n#7\n")

    '''
    7. Transformada de Coseno Discreta (DCT).
    7.1. DCT nos canais completos
    7.1.1. Crie uma função para calcular a DCT de um canal completo. Utilize a função scipy.fftpack.dct.
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
    7.1.4. Decoder: Aplique a função inversa (7.1.2) e certifique-se de que consegue obter os valores originais de Y_d, Cb_d e Cr_d. 
    '''

    Y_d, Cb_d, Cr_d= decoder(None,None,None,None,None,None,False,False,False,False, Y_d_dct,Cb_d_dct,Cr_d_dct,None,True)

    print("Valores originais Y_d Cb_d Cr_d, pós DCT:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)

    '''
    7.2.1. Usando as mesmas funções para cálculo da DCT, crie uma função que calcule a
        DCT de um canal completo em blocos BSxBS.
    7.2.2. Crie também a função inversa (IDCT BSxBS).
    7.2.3. Encoder: Aplique a função desenvolvida (7.2.1) a Y_d, Cb_d, Cr_d com blocos 8x8
        e visualize as imagens obtidas (Y_dct8, Cb_dct8, Cr_dct8).
    7.2.4. Decoder: Aplique a função inversa (7.2.2) e certifique-se de que consegue obter
        os valores originais de Y_d, Cb_d e Cr_d.
    '''

    Y_d_dct8, Cb_d_dct8, Cr_d_dct8=encoder(None,False,False,False,False,Y_d,Cb_d,Cr_d,None,None,False,True,8)

    Y_d, Cb_d, Cr_d = decoder(None,None,None,None,None,None,False,False,False,False, Y_d_dct8, Cb_d_dct8, Cr_d_dct8,None,False,True,8)

    print("\nValores originais Y_d Cb_d Cr_d, pós DCT com blocos 8x8:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)

    #7.3. DCT em blocos 64x64.
    Y_d_dct64, Cb_d_dct64, Cr_d_dct64=encoder(None,False,False,False,False,Y_d,Cb_d,Cr_d,None,None,False,True,64)
    Y_d, Cb_d, Cr_d=decoder(None,None,None,None,None,None,False,False,False,False,Y_d_dct64, Cb_d_dct64, Cr_d_dct64,None,False,True,64)

    print("\nValores originais Y_d Cb_d Cr_d, pós DCT com blocos 64x64:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)
    
    '''
    8. Quantização.
    8.1. Crie uma função para quantizar os coeficientes da DCT para cada bloco 8x8. 
    8.3. Encoder: Quantize os coeficientes da DCT, usando os seguintes factores de qualidade:
    10, 25, 50, 75 e 100. Visualize as imagens obtidas (Y_q, CB_q e Cr_q).
    '''

    matriz_quantizacao_Y = np.array([
      [16, 11, 10, 16, 24, 40, 51, 61],
      [12, 12, 14, 19, 26, 58, 60, 55],
      [14, 13, 16, 24, 40, 57, 69, 56],
      [14, 17, 22, 29, 51, 87, 80, 62],
      [18, 22, 37, 56, 68, 109, 103, 77],
      [24, 35, 55, 64, 81, 104, 113, 92],
      [49, 64, 78, 87, 103, 121, 120, 101],
      [72, 92, 95, 98, 112, 100, 103, 99]
      ],dtype=np.uint16)

    matriz_quantizacao_CbCr = np.array([
      [17, 18, 24, 47, 99, 99, 99, 99],
      [18, 21, 26, 66, 99, 99, 99, 99],
      [24, 26, 56, 99, 99, 99, 99, 99],
      [47, 66, 99, 99, 99, 99, 99, 99],
      [99, 99, 99, 99, 99, 99, 99, 99],
      [99, 99, 99, 99, 99, 99, 99, 99],
      [99, 99, 99, 99, 99, 99, 99, 99],
      [99, 99, 99, 99, 99, 99, 99, 99]
      ],dtype=np.uint16)
    
    qualidade = 10

    Y_d_dct8_quant, Cb_d_dct8_quant, Cr_d_dct8_quant = encoder(None,False,False,False,False,Y_d_dct8, Cb_d_dct8, Cr_d_dct8,None,None,False,False,8,True,qualidade,matriz_quantizacao_Y,matriz_quantizacao_CbCr)
    
    Y_d_dct8, Cb_d_dct8, Cr_d_dct8 = decoder(None,None,None,None,None,None,False,False,False,False,Y_d_dct8_quant, Cb_d_dct8_quant, Cr_d_dct8_quant,None,False,False,8,True,matriz_quantizacao_Y,matriz_quantizacao_CbCr,qualidade)
    
    '''
    9. Codificação DPCM dos coeficientes DC.
    9.3. Encoder: Aplique a função 9.1 aos valores da DCT quantizada, obtendo Y_dpcm,
    Cb_dpcm e Cr_dpcm). 
    '''
       
    Y_d_dct8_quant_dpcm,Cb_d_dct8_quant_dpcm,Cr_d_dct8_quant_dpcm = encoder(None, False, False, False, False,Y_d_dct8_quant, Cb_d_dct8_quant, Cr_d_dct8_quant, None, None, False, False, None, None, None, None, None, True, None)
    
    dpcm = [Y_d_dct8_quant_dpcm,Cb_d_dct8_quant_dpcm,Cr_d_dct8_quant_dpcm]
    bruh = mult_DCT_log(dpcm)
    display_images([bruh[0], bruh[1], bruh[2]], ['DPCM Y', 'DPCM Cb', 'DPCM Cr'])
  
    '''
    9.4. Decoder: Aplique a função inversa (9.2) e certifique-se de que consegue obter os
    valores originais de Y_q, Cb_q e Cr_q.
    '''

    Y_d_dct8_quant,Cb_d_dct8_quant,Cr_d_dct8_quant = decoder(None,None,None,None,None,None,False,False,False,False,None, None, None,None,False,False,None,False,None,None,None,True,dpcm)
    invdpcm = [Y_d_dct8_quant,Cb_d_dct8_quant,Cr_d_dct8_quant]
    bruh = mult_DCT_log(invdpcm)
    display_images([bruh[0], bruh[1], bruh[2]], ['Inverse DPCM Y', 'Inverse DPCM Cb', 'Inverse DPCM Cr'])

    print("\n#9\n")
    print("Valores originais Y_d Cb_d Cr_d, após inverter o dpcm :")
    print("Dimensões de Y_d",invdpcm[0].shape)
    print("Dimensões de Cb_d",invdpcm[1].shape)
    print("Dimensões de Cr_d",invdpcm[2].shape)

    '''
    10. Codificação e descodificação end-to-end.
    10.1. Encoder: Codifique as imagens fornecidas com os seguintes parâmetros de qualidade:
    10, 25, 50, 75 e 100.
    10.2. Decoder: Reconstrua as imagens com base no resultado de 10.1.
    '''

    og = (h,w)
    img_reconstr = decoder(None,None,None,None,None,og,False,False,False,False,Y_d_dct8_quant_dpcm,Cb_d_dct8_quant_dpcm,Cr_d_dct8_quant_dpcm, cv2.INTER_LINEAR,False,False,8,False,matriz_quantizacao_Y,matriz_quantizacao_CbCr,qualidade,False,None,True)    

    '''
    10.3. Crie uma função para cálculo da imagem das diferenças (entre o canal Y da original e da descompactada).
    '''

    channel_diference(Y_og_ex10,img_reconstr)

    '''
    10.4. Crie uma função para calculo das métricas de distorção MSE, RMSE, SNR, PSNR,
    max_diff e avg_diff (por comparação da imagem original com a descompactada).
    '''

    print("\n#10")

    img_og = plt.imread(fname)
    
    padded_img_r, (h, w) = padding(img_reconstr)

    R,G,B = splitRGB(padded_img_r)
    
    Y_r, Cb_r, Cr_r = rgb_to_ycbcr(R,G,B)
    
    distorcao(img_og,img_reconstr,Y_og_ex10,Y_r)

    return

if __name__ == "__main__":
    main()
