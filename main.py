import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from scipy.fftpack import dct, idct
import cv2

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


def encoder(img = None, pad=False, split=False, RGB_to_YCBCR=False, sub=False, Y=None, Cb=None, Cr=None, subsampling_type=None, interpolation=None,dct = False,dctBlocks=False,step=None,quant = False,fq = None,quant_matrix_Y = None,quant_matrix_CbCr = None,dpcmfds=False,channel=None):

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
  
  elif dpcmfds:
     return DPCM(channel)

     
 
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
  


def newCmap(keyColors = [(0,0,0),(1,1,1)], name = "gray", N= 256):
    cm = clr.LinearSegmentedColormap.from_list(name, keyColors, N)
    return cm


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


def padding(img):
  # Captura a altura (h) e a largura (ln) da imagem.
  h,w = img.shape[:2]
  p1, p2, p3 = splitRGB(img)

  # Obtém as dimensões do primeiro canal de cor.
  r,c = p1.shape

  v_pad = 32 - (r % 32) if r % 32 > 0 else 0
  h_pad = 32 - (c % 32) if c % 32 > 0 else 0  


  p1 = np.vstack([p1, np.repeat(np.array(p1[-1,:], ndmin = 2), v_pad, axis=0)])
  p1 = np.hstack([p1, np.repeat(np.array(p1[:,-1], ndmin = 2), h_pad, axis=0).T])

  p2 = np.vstack([p2, np.repeat(np.array(p2[-1,:], ndmin = 2), v_pad, axis=0)])
  p2 = np.hstack([p2, np.repeat(np.array(p2[:,-1], ndmin = 2), h_pad, axis=0).T])

  p3 = np.vstack([p3, np.repeat(np.array(p3[-1,:], ndmin = 2), v_pad, axis=0)])
  p3 = np.hstack([p3, np.repeat(np.array(p3[:,-1], ndmin = 2), h_pad, axis=0).T])

  # Empilha os três canais de volta em uma imagem e retorna junto com as dimensões originais.
  return np.dstack((p1, p2, p3)), (h,w)

def unpadding(img, og):
  return img[:og[0], :og[1], :]

#5.3
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

#5.4
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

#6.1
def sub_amostragem(Y, Cb, Cr, subsampling_type, interpolation):
      
  width, height = Y.shape[1], Y.shape[0]

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

#7.1.3
def DCT(Y, Cb, Cr):
    # Applying DCT
    Y_dct = dct(dct(Y, norm='ortho').T, norm='ortho').T
    Cb_dct = dct(dct(Cb, norm='ortho').T, norm='ortho').T
    Cr_dct = dct(dct(Cr, norm='ortho').T, norm='ortho').T
    
    # Log transformation for better visualization
    Y_dct_log = np.log(np.abs(Y_dct) + 0.0001)
    Cb_dct_log = np.log(np.abs(Cb_dct) + 0.0001)
    Cr_dct_log = np.log(np.abs(Cr_dct) + 0.0001)

     # Using display_images function for displaying DCT images
    display_images([Y_dct_log, Cb_dct_log, Cr_dct_log], ['Log DCT of Y', 'Log DCT of Cb', 'Log DCT of Cr'], colormap='gray')

    return Y_dct,Cb_dct,Cr_dct

def single_DCT_log(channel):
   return np.log(np.abs(channel) + .0001)

def mult_DCT_log(channels):
  return single_DCT_log(channels[0]),single_DCT_log(channels[1]),single_DCT_log(channels[2])

#7.1.4
def invertDCT(Y_dct, Cb_dct, Cr_dct):  
    # Applying IDCT
    Y_inv_dct = idct(idct(Y_dct, norm='ortho').T, norm='ortho').T
    Cb_inv_dct = idct(idct(Cb_dct, norm='ortho').T, norm='ortho').T
    Cr_inv_dct = idct(idct(Cr_dct, norm='ortho').T, norm='ortho').T

    # Using display_images function for displaying inverse DCT images
    display_images([Y_inv_dct, Cb_inv_dct, Cr_inv_dct], 
                   ['Inverse DCT of Y', 'Inverse DCT of Cb', 'Inverse DCT of Cr'], 
                   colormap='gray')


    return Y_inv_dct,Cb_inv_dct,Cr_inv_dct

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
#7.2.3
def DCT_Blocks(Y, Cb, Cr, tamanho_bloco=8):  
    Y_dct = DCT_Blocks_Channel(Y, tamanho_bloco)
    Cb_dct = DCT_Blocks_Channel(Cb, tamanho_bloco)
    Cr_dct = DCT_Blocks_Channel(Cr, tamanho_bloco)

    Y_dct_log = np.log(np.abs(Y_dct) + 0.0001)
    Cb_dct_log = np.log(np.abs(Cb_dct) + 0.0001)
    Cr_dct_log = np.log(np.abs(Cr_dct) + 0.0001)

  
    display_images([Y_dct_log, Cb_dct_log, Cr_dct_log], 
                   ['DCT {}x{} of Y'.format(tamanho_bloco, tamanho_bloco), 
                    'DCT {}x{} of Cb'.format(tamanho_bloco, tamanho_bloco), 
                    'DCT {}x{} of Cr'.format(tamanho_bloco, tamanho_bloco)], 
                   colormap='gray')
    
    return Y_dct, Cb_dct, Cr_dct


def single_DCT_log(channel):
   return np.log(np.abs(channel) + .0001)

def mult_DCT_log(channels):
  return single_DCT_log(channels[0]),single_DCT_log(channels[1]),single_DCT_log(channels[2])

#7.2.4
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

  
  display_images([idctOut_Y, idctOut_Cb, idctOut_Cr], 
                   ['Inverse DCT of Y', 'Inverse DCT of Cb', 'Inverse DCT of Cr'], 
                   colormap='gray')

  return idctOut_Y,idctOut_Cb,idctOut_Cr


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

  return Qs_Cro, Qs_Lum
     

def quantize_block(dct_block, quant_matrix):
    
    quantized_block = np.round(dct_block / quant_matrix)

    return quantized_block

#8.3
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

  # Using display_images function for displaying quantized DCT images
  display_images([Y_dct_log, Cb_dct_log, Cr_dct_log], [
        'DCT quant {} {}x{} of Y'.format(fq, step, step), 
        'DCT quant {} {}x{} of Cb'.format(fq, step, step), 
        'DCT quant {} {}x{} of Cr'.format(fq, step, step)
    ], colormap='gray')


  return Y_dct, Cb_dct, Cr_dct

def dequantize_block(quantized_block, fq,Lum_quant_matrix_std,Cro_quant_matrix_std,Crominancia = False,Luminancia = False):

  Qs_Cro, Qs_Lum = adj_quant_matrix(fq,Lum_quant_matrix_std,Cro_quant_matrix_std)

  if Crominancia==True:
     dequantized_block = quantized_block * Qs_Cro
  
  elif Luminancia == True:
     dequantized_block = quantized_block * Qs_Lum
     
  return dequantized_block

#8.4
def dequantized_dct(Y_dct_quant, Cb_dct_quant, Cr_dct_quant,quant_matrix_Y,quant_matrix_CbCr,step, fq):

    lines_Y, cols_Y = Y_dct_quant.shape
    lines_Cb, cols_Cb = Cb_dct_quant.shape
    lines_Cr, cols_Cr = Cr_dct_quant.shape

    canal_Y_dct_desquantizado = np.zeros_like(Y_dct_quant)

    canal_Cb_dct_desquantizado = np.zeros_like(Cb_dct_quant)

    canal_Cr_dct_desquantizado = np.zeros_like(Cr_dct_quant)

    # Process Y channel
    for i in range(0, lines_Y, step):
        for j in range(0, cols_Y, step):
            quant_dct_block = Y_dct_quant[i:i+step, j:j+step]
            canal_Y_dct_desquantizado[i:i+step, j:j+step] = dequantize_block(quant_dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=False, Luminancia=True)

    # Process Cb channel
    for i in range(0, lines_Cb, step):
        for j in range(0, cols_Cb, step):
            quant_dct_block = Cb_dct_quant[i:i+step, j:j+step]
            canal_Cb_dct_desquantizado[i:i+step, j:j+step] = dequantize_block(quant_dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=True, Luminancia=False)

    # Process Cr channel
    for i in range(0, lines_Cr, step):
        for j in range(0, cols_Cr, step):
            quant_dct_block = Cr_dct_quant[i:i+step, j:j+step]
            canal_Cr_dct_desquantizado[i:i+step, j:j+step] = dequantize_block(quant_dct_block, fq, quant_matrix_Y, quant_matrix_CbCr, Crominancia=True, Luminancia=False)

    Y_dct_log = np.log(np.abs(canal_Y_dct_desquantizado) + 0.0001)
    Cb_dct_log = np.log(np.abs(canal_Cb_dct_desquantizado) + 0.0001)
    Cr_dct_log = np.log(np.abs(canal_Cr_dct_desquantizado) + 0.0001)

   
    display_images([Y_dct_log, Cb_dct_log, Cr_dct_log], [
        'DCT dequant {} {}x{} of Y'.format(fq, step, step), 
        'DCT dequant {} {}x{} of Cb'.format(fq, step, step), 
        'DCT dequant {} {}x{} of Cr'.format(fq, step, step)
    ], colormap='gray')

    return canal_Y_dct_desquantizado, canal_Cb_dct_desquantizado, canal_Cr_dct_desquantizado

#9.3
def DPCM(channels):
  Y_dct_log = channels[0].copy()
  Cb_dct_log = channels[1].copy()
  Cr_dct_log = channels[2].copy()

  y, cb, cr = channels

  original_Y = Y_dct_log.shape
  original_Cb = Cb_dct_log.shape

  dc0 = [0,0,0]
  for i in range(0, original_Y[0], 8):
    for j in range(0, original_Y[1], 8):
      if i == 0 and j == 0:
        dc0 = [y[0,0], cb[0,0], cr[0,0]]
        continue
      if i < original_Cb[0] and j < original_Cb[1]:
        dc = cb[i,j], cr[i,j]
        diff = dc[0] - dc0[1], dc[1] - dc0[2]
        Cb_dct_log[i,j], Cr_dct_log[i,j] = diff
        dc0[1] = dc[0]
        dc0[2] = dc[1]
      dc = y[i,j]
      diff = dc - dc0[0]
      Y_dct_log[i,j] = diff
      dc0[0] = dc

  return Y_dct_log, Cb_dct_log, Cr_dct_log

#9.4
def invDPCM(channels):
  Y_dct_log = channels[0].copy()
  Cb_dct_log = channels[1].copy()
  Cr_dct_log = channels[2].copy()

  y, cb, cr = channels

  original_Y = Y_dct_log.shape
  original_Cb = Cb_dct_log.shape

  dc0 = [0,0,0]
  for i in range(0, original_Y[0], 8):
    for j in range(0, original_Y[1], 8):
      if i == 0 and j == 0:
        dc0 = [y[0,0], cb[0,0], cr[0,0]]
        continue
      if i < original_Cb[0] and j < original_Cb[1]:
        dc = cb[i,j], cr[i,j]
        diff = dc[0] + dc0[1], dc[1] + dc0[2]
        Cb_dct_log[i,j], Cr_dct_log[i,j] = diff
        dc0[1] = diff[0]
        dc0[2] = diff[1]
      dc = y[i,j]
      diff = dc + dc0[0]
      Y_dct_log[i,j] = diff
      dc0[0] = diff
  return y, cb, cr


def display_images(images, titles, colormap='gray'):
    
    plt.figure(figsize=(12, 4))
    num_images = len(images)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap=colormap)
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()
  

def main():
    
    #3.1
    fname = "airport.bmp"
    img = plt.imread(fname)
    
    original_pixel = img[0, 0]
   
   #3.2
    cm_red=newCmap([(0,0,0),(1,0,0)], "cm_red", 256)
    cm_green=newCmap([(0,0,0),(0,1,0)], "cm_green", 256)
    cm_blue=newCmap([(0,0,0),(0,0,1)], "cm_blue", 256)
    
    #3.4
    showImg(img,fname,"Imagem original: ")
    print("\n#4\n")
    print("Dimensão Original: " + str(img.shape))
  
    R, G, B = encoder(img,pad=False,split=True)

    decoder(R, G, B, img_ycbcr = None,og = None,unpad = False, join= True,YCBCR_to_RGB = False)
    #3.6
    showImg(R,fname,"Img Red: ",cm_red)
    showImg(G,fname,"Img Green: ",cm_green)
    showImg(B,fname,"Img Blue: ",cm_blue)

    #4.1
    padded_img, (h, w) = encoder(img, pad=True, split=False)
    print("Dimensão Padded: "+ str(padded_img.shape))  

    R,G,B = splitRGB(padded_img)
    showImg(R,fname,"Img Red padded: ",cm_red)
    showImg(G,fname,"Img Green padded: ",cm_green)
    showImg(B,fname,"Img Blue padded : ",cm_blue)

    #4.2
    unpadded_img = decoder(R,G,B,img_ycbcr = None,padded_img = padded_img, og = (h,w),unpad = True,join = False,YCBCR_to_RGB = False)
    print("Dimensão Unpadded: " + str(unpadded_img.shape))  
    
    #5.1
    y,cb,cr = encoder(padded_img,pad = False,split = False, RGB_to_YCBCR = True)
    
    showImg(y,fname,'Canal Y (Luminância)','gray')
    
    showImg(cb,fname,'Canal Cb (Diferença de Azul)','gray')
    
    showImg(cr,fname,'Canal Cr (Diferença de Vermelho)','gray')
    
    encoded_ycbcr_img = np.dstack((y, cb, cr))

    #5.2
    recovered_img = decoder(R,G,B,encoded_ycbcr_img,padded_img = padded_img, og = (h,w),unpad = False,join = False,YCBCR_to_RGB = True)


    #5.4
    recovered_pixel = recovered_img[0, 0]
    
    splitRGB(recovered_img)

    print(f'Original RGB pixel [0,0]: {original_pixel}')
    print(f'Recovered RGB pixel [0,0]: {recovered_pixel}')
  

    #6.1
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False, True, y, cb, cr, "4:2:0", cv2.INTER_CUBIC)
    display_images([Y_d, Cb_d, Cr_d], ['Y downsampling 4:2:0 (CUBIC)', 'Cb downsampling 4:2:0 (CUBIC)', 'Cr downsampling 4:2:0 (CUBIC)'])

    #6.2
    Y, Cb, Cr = decoder(None, None, None, None, None, None, False, False, False, True, Y_d, Cb_d, Cr_d, cv2.INTER_CUBIC)
    display_images([Y, Cb, Cr], ['Y upsampling 4:2:0 (CUBIC)', 'Cb upsampling 4:2:0 (CUBIC)', 'Cr upsampling 4:2:0 (CUBIC)'])

    #6.1
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False, True, y, cb, cr, "4:2:2", cv2.INTER_CUBIC)
    display_images([Y_d, Cb_d, Cr_d], ['Y downsampling 4:2:2 (CUBIC)', 'Cb downsampling 4:2:2 (CUBIC)', 'Cr downsampling 4:2:2 (CUBIC)'])

    #6.2
    Y, Cb, Cr = decoder(None, None, None, None, None, None, False, False, False, True, Y_d, Cb_d, Cr_d, cv2.INTER_CUBIC)
    display_images([Y, Cb, Cr], ['Y upsampling 4:2:2 (CUBIC)', 'Cb upsampling 4:2:2 (CUBIC)', 'Cr upsampling 4:2:2 (CUBIC)'])
     
    #6.1
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False, True, y, cb, cr, "4:2:0", cv2.INTER_LINEAR)
    display_images([Y_d, Cb_d, Cr_d], ['Y downsampling 4:2:0 (LINEAR)', 'Cb downsampling 4:2:0 (LINEAR)', 'Cr downsampling 4:2:0 (LINEAR)'])

    print("---[downsampling 4:2:0 (LINEAR)]---\n")
    print("Dimensões de Y_d:", Y_d.shape)
    print("Dimensões de Cb_d:", Cb_d.shape)
    print("Dimensões de Cr_d:", Cr_d.shape)

    #6.2
    Y, Cb, Cr = decoder(None, None, None, None, None, None, False, False, False, True, Y_d, Cb_d, Cr_d, cv2.INTER_LINEAR)
    display_images([Y, Cb, Cr], ['Y upsampling 4:2:0 (LINEAR)', 'Cb upsampling 4:2:0 (LINEAR)', 'Cr upsampling 4:2:0 (LINEAR)'])

    #6.1
    Y_d, Cb_d, Cr_d = encoder(padded_img, False, False, False, True, y, cb, cr, "4:2:2", cv2.INTER_LINEAR)
    display_images([Y_d, Cb_d, Cr_d], ['Y downsampling 4:2:2 (LINEAR)', 'Cb downsampling 4:2:2 (LINEAR)', 'Cr downsampling 4:2:2 (LINEAR)'])

    #6.2
    Y, Cb, Cr = decoder(None, None, None, None, None, None, False, False, False, True, Y_d, Cb_d, Cr_d, cv2.INTER_LINEAR)
    display_images([Y, Cb, Cr], ['Y upsampling 4:2:2 (LINEAR)', 'Cb upsampling 4:2:2 (LINEAR)', 'Cr upsampling 4:2:2 (LINEAR)'])

    #7.1.1
    Y_d_dct, Cb_d_dct, Cr_d_dct = encoder(None,False,False,False,False,Y_d,Cb_d,Cr_d,None,None,True)
     
    #7.1.2
    Y_d, Cb_d, Cr_d= decoder(None,None,None,None,None,None,False,False,False,False, Y_d_dct,Cb_d_dct,Cr_d_dct,None,True)

    print("Valores originais Y_d Cb_d Cr_d, pós DCT:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)

    #7.2.1
    Y_dct8, Cb_dct8, Cr_dct8=encoder(None,False,False,False,False,Y_d,Cb_d,Cr_d,None,None,False,True,8)
    #7.2.2
    Y_d, Cb_d, Cr_d=decoder(None,None,None,None,None,None,False,False,False,False, Y_dct8, Cb_dct8, Cr_dct8,None,False,True,8)

    print("\nValores originais Y_d Cb_d Cr_d, pós DCT com blocos 8x8:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)

    #7.2.1 64
    Y_dct64, Cb_dct64, Cr_dct64=encoder(None,False,False,False,False,Y_d,Cb_d,Cr_d,None,None,False,True,64)

    #7.2.2 64
    Y_d, Cb_d, Cr_d=decoder(None,None,None,None,None,None,False,False,False,False, Y_dct64, Cb_dct64, Cr_dct64,None,False,True,64)

    print("\nValores originais Y_d Cb_d Cr_d, pós DCT com blocos 64x64:")
    print("Dimensões de Y_d",Y_d.shape)
    print("Dimensões de Cb_d",Cb_d.shape)
    print("Dimensões de Cr_d",Cr_d.shape)
   
    #8.1
    Y_dct8_quant, Cb_dct8_quant, Cr_dct8_quant = encoder(None,False,False,False,False,Y_dct8, Cb_dct8, Cr_dct8,None,None,False,False,8,True,50,matriz_quantizacao_Y,matriz_quantizacao_CbCr)
    Y_DPCM,CB_DPCM,CR_DPCM= Y_dct8_quant.copy(), Cb_dct8_quant.copy(), Cr_dct8_quant.copy()
    channel=[Y_DPCM,CB_DPCM,CR_DPCM]
    
    #8.2
    dequantized_dct(Y_dct8_quant, Cb_dct8_quant, Cr_dct8_quant,matriz_quantizacao_Y,matriz_quantizacao_CbCr,8, 50)
   
    #9.1
    dpcm = encoder(None, False, False, False, False, Y_DPCM, CB_DPCM, CR_DPCM, None, None, False, False, None, None, None, None, None, True, channel)
    bruh = mult_DCT_log(dpcm)
    display_images([bruh[0], bruh[1], bruh[2]], ['DPCM Y', 'DPCM Cb', 'DPCM Cr'])
   

   #9.2
    invdpcm = invDPCM(dpcm)
    bruh = mult_DCT_log(invdpcm)
    display_images([bruh[0], bruh[1], bruh[2]], ['Inverse DPCM Y', 'Inverse DPCM Cb', 'Inverse DPCM Cr'])

    conta = [invdpcm[0] - dpcm[0], invdpcm[1] - dpcm[1], invdpcm[2] - dpcm[2]]
    print(conta)
      
    return

if __name__ == "__main__":
    main()