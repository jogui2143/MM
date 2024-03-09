#!/usr/bin/env python
# coding: utf-8

# In[47]:


import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import dct, idct


Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
],dtype=np.uint16)

Q_CbCr = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
],dtype=np.uint16)

qualidade=75

# In[48]:


def encoder(img):
    """
    Separa a imagem em componentes RGB e aplica padding, converte para YCbCr
    """
    
    # 3.2 e 3.3. Visualização com colormap personalizado
    cm_gray = newCmap([(0,0,0), (1,1,1)], "gray", 256)
    cm_red = newCmap([(0,0,0),(1,0,0)], "cm_red", 256)
    cm_green = newCmap([(0,0,0),(0,1,0)], "cm_green", 256)
    cm_blue = newCmap([(0,0,0),(0,0,1)], "cm_blue", 256)
    
    # Separa a imagem nos canais
    R, G, B = splitRGB(img)
    
    # Aplica padding a cada canal  % 32
    R_padded = add_padding(R)
    G_padded = add_padding(G)
    B_padded = add_padding(B)
    
    # Visualização da imagem RGB com padding 
    imgPadded = joinRGB(R_padded, G_padded, B_padded)
    showImg(imgPadded, "Imagem RGB com Padding: ", cmap=cm_gray)
    
    #Converte os canais RGB com padding para YCbCr
    Y,Cb,Cr = rgb_to_ycbcr(R_padded, G_padded, B_padded)
    
    # 5.3 Visualizar canais YCbCr
    #showImg(Y, "Canal Y", cmap=cm_gray)
    #showImg(Cb, "Canal Cb", cmap=cm_gray)
    #showImg(Cr, "Canal Cr", cmap=cm_gray)
    
    
    Y_d, Cb_d, Cr_d = subamostrar(Y, Cb, Cr, 0.5,1, metodo=cv2.INTER_LINEAR)

    
    print("Checkpoint OK (Y_d,Cb_d,Cr_d)")
    print(Y_d[8:16,8:16])
    print(Cb_d[8:16,8:16])
    print(Cr_d[8:16,8:16])
    
    
    # Visualize os canais com downsampling
    showImg(Y_d, "Canal Y Sub-amostrado", cmap='gray')
    showImg(Cb_d, "Canal Cb Sub-amostrado 4:2:2", cmap='gray')
    showImg(Cr_d, "Canal Cr Sub-amostrado 4:2:2", cmap='gray')
    
    # Apresente as dimensões das matrizes correspondentes
    print("Dimensões Y_d:", Y_d.shape)
    print("Dimensões Cb_d:", Cb_d.shape)
    print("Dimensões Cr_d:", Cr_d.shape)
    
    
    #7.1
   # Aplicar DCT y,cb,cr = encoder(file)
    Y_dct = calcular_dct_canal(Y_d)
    Cb_dct = calcular_dct_canal(Cb_d)
    Cr_dct = calcular_dct_canal(Cr_d)
    
    # Visualizar DCT
    visualizar_canal_dct(Y_dct, "Y DCT")
    visualizar_canal_dct(Cb_dct, "Cb DCT")
    visualizar_canal_dct(Cr_dct, "Cr DCT")
    
    # 7.2 e 7.3: Aplicar DCT em blocos de 8x8
    Y_dct8 = aplicar_dct_blocos(Y_d, 8)
    Cb_dct8 = aplicar_dct_blocos(Cb_d, 8)
    Cr_dct8 = aplicar_dct_blocos(Cr_d, 8)

    '''
    print("Checkpoint OK (Y_d_dct8,Cb_d_dct8,Cr_d_dct8)")
    print(Y_dct8[8:16,8:16])
    print(Cb_dct8[8:16,8:16])
    print(Cr_dct8[8:16,8:16])
    '''
    
    # Visualização de DCT 8x8
    visualizar_canal_dct(Y_dct8, "Y DCT 8x8")
    visualizar_canal_dct(Cb_dct8, "Cb DCT 8x8")
    visualizar_canal_dct(Cr_dct8, "Cr DCT 8x8")
    
    Y_dct64 = aplicar_dct_blocos(Y_d,64)
    Cb_dct64 = aplicar_dct_blocos(Cb_d, 64)
    Cr_dct64 = aplicar_dct_blocos(Cr_d, 64)
    
    visualizar_canal_dct(Y_dct64, "Y DCT 64x64")
    visualizar_canal_dct(Cb_dct64, "Cb DCT 64x64")
    visualizar_canal_dct(Cr_dct64, "Cr DCT 64x64")
    
    
    Y_quantizado = quantizar_blocos_dct(Y_dct8,Q_Y,qualidade)
    Cb_quantizado = quantizar_blocos_dct(Cb_dct8, Q_CbCr,qualidade)
    Cr_quantizado = quantizar_blocos_dct(Cr_dct8, Q_CbCr,qualidade)
    visualizar_canal_dct(Y_quantizado, "Y Quantizado")
    visualizar_canal_dct(Cb_quantizado, "Cb Quantizado")
    visualizar_canal_dct(Cr_quantizado, "Cr Quantizado")
    #print("Y_Q")
    #print(Y_quantizado[8:16,8:16])

    
    print("Checkpoint OK (Y_d_dct8_quant, Cb_d_dct8_quant, Cr_d_dct8_quant)")
    print(Y_quantizado[8:16,8:16])
    print(Cb_quantizado[8:16,8:16])
    print(Cr_quantizado[8:16,8:16])
    
    

    Y_DPCM=codificar_dpcm_dc_imagem(Y_quantizado)
    Cb_DPCM=codificar_dpcm_dc_imagem(Cb_quantizado)
    Cr_DPCM=codificar_dpcm_dc_imagem(Cr_quantizado)

    '''
    print("Checkpoint OK (Y_DPCM, Cb_DPCM, Cr_DPCM)")
    print(Y_DPCM[8:16,8:16])
    print(Cb_DPCM[8:16,8:16])
    print(Cr_DPCM[8:16,8:16])
    '''
    
    visualizar_canal_dct(Y_DPCM, "Y_DPCM")
    visualizar_canal_dct(Cb_DPCM, "Cb_DPCM")
    visualizar_canal_dct(Cr_DPCM, "Cr_DPCM")

    #print("Y_DPCM")
    #print(Y_DPCM[8:16,8:16])
    
    #Retorna os canais RGB com padding , os Y Cb Cr e as dimensoes originais 
    return  Y_DPCM,Cb_DPCM,Cr_DPCM,img.shape[:2],Y


# In[49]:


def decoder(Y_DPCM,Cb_DPCM,Cr_DPCM,original_shape):
    """
    Converte YCbCr p/ RGB, remove padding, combina os canais RGB para obter a imagem original
    """
    # 3.2 e 3.3. Visualização com colormap personalizado
    cm_gray = newCmap([(0,0,0), (1,1,1)], "gray", 256)
    cm_red = newCmap([(0,0,0),(1,0,0)], "cm_red", 256)
    cm_green = newCmap([(0,0,0),(0,1,0)], "cm_green", 256)
    cm_blue = newCmap([(0,0,0),(0,0,1)], "cm_blue", 256)
    
    Y_quantizado= descodificar_dpcm_dc_imagem(Y_DPCM)
    Cb_quantizado = descodificar_dpcm_dc_imagem(Cb_DPCM)
    Cr_quantizado = descodificar_dpcm_dc_imagem(Cr_DPCM)
    visualizar_canal_dct(Y_quantizado, "Y Quantizado")
    visualizar_canal_dct(Cb_quantizado, "Cb Quantizado")
    visualizar_canal_dct(Cr_quantizado, "Cr Quantizado")
    print("Y_iDPCM")
    print(Y_quantizado[8:16,8:16])
    
    
    # 8.4 Desquantização
    Y_dct8= desquantizar_blocos(Y_quantizado, Q_Y, qualidade)
    Cb_dct8 = desquantizar_blocos(Cb_quantizado, Q_CbCr,qualidade)
    Cr_dct8 = desquantizar_blocos(Cr_quantizado, Q_CbCr,qualidade)
    visualizar_canal_dct(Y_dct8, "Y Desquantizado (Blocos) ")
    visualizar_canal_dct(Cb_dct8, "Cb Desquantizado (Blocos)")
    visualizar_canal_dct(Cr_dct8, "Cr Desquantizado (Blocos)")
    
    # 7.2.4 Aplicar IDCT aos blocos (opcional)
    Y_rec = aplicar_idct_blocos(Y_dct8,8)
    Cb_rec = aplicar_idct_blocos(Cb_dct8,8)
    Cr_rec = aplicar_idct_blocos(Cr_dct8,8)
    showImg(Y_rec, "Canal Y_d Reconstruído", cmap='gray')
    showImg(Cb_rec, "Canal Y_d Reconstruído", cmap='gray')
    showImg(Cr_rec, "Canal Y_d Reconstruído", cmap='gray')
    
    
    #7.1.4 Aplicar IDCT à DCT
    #Y_rec = calcular_idct_canal(Y_dct)
    #Cb_rec = calcular_idct_canal(Cb_dct)
    #Cr_rec = calcular_idct_canal(Cr_dct)
    #showImg(Y_rec, "Canal Y_d Reconstruído", cmap='gray')
    #showImg(Cb_rec, "Canal Y_d Reconstruído", cmap='gray')
    #showImg(Cr_rec, "Canal Y_d Reconstruído", cmap='gray')
    
    # 6.4 Decoder: Upsampling e reconstrução dos canais Y, Cb e Cr
    Y_up, Cb_up, Cr_up = superamostrar(Y_rec, Cb_rec, Cr_rec, metodo=cv2.INTER_LINEAR)
    
    # Visualize os canais reconstruídos
    showImg(Y_up, "Canal Y Reconstruído (Upsampling)", cmap='gray')
    showImg(Cb_up, "Canal Cb Reconstruído (Upsampling)", cmap='gray')
    showImg(Cr_up, "Canal Cr Reconstruído (Upsampling)", cmap='gray')
    
    # 5.4. Decoder: Recuperar RGB de YCbCr
    R_rec, G_rec, B_rec = ycbcr_to_rgb(Y_up, Cb_up, Cr_up)
    imgRGB_from_YCbCr = joinRGB(R_rec, G_rec, B_rec)
    showImg(imgRGB_from_YCbCr, "Imagem de YCbCr para RGB (Com Padding)")
    
    #Converte YCbCr para RGB
    R_padded,G_padded,B_padded=ycbcr_to_rgb(Y_up, Cb_up, Cr_up)
    
    # Remove padding
    R = remove_padding(R_padded, original_shape)
    G = remove_padding(G_padded, original_shape)
    B = remove_padding(B_padded, original_shape)
    
    # Junta os canais
    imgRec = joinRGB(R, G, B)
    showImg(imgRec, "Imagem Reconstruída: ", cmap=cm_gray)
    
    return Y_up,imgRec


# In[50]:


#3.2
def newCmap(keyColors = [(0,0,0),(1,1,1)], name = "gray", N= 256):
    """
    Esquema de cor - cinzento 
    """
    cm = clr.LinearSegmentedColormap.from_list(name, keyColors, N)
    return cm


# In[51]:


#3.3
def showImg(img, fname="", caption="", cmap=None):
    #print(img.shape)
    #print(img.dtype)
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(f"{caption}{fname}\nDimensões: {img.shape[0]}x{img.shape[1]}")
    plt.show()


# In[52]:


#3.4
def splitRGB(img):
    #Separa nos componentes | Acede a todas as linhas , todas as colunas do canal de cor
    R = img[:,:,0] #  do canal 0 - Red , 1 Green , 2 Blue 
    G = img[:,:,1]
    B = img[:,:,2]
    return R, G, B


# In[53]:


#3.5
def joinRGB(R,G,B):
    """
    Combina três canais de cor R, G e B em uma única imagem RGB.
    """
    nl,nc=R.shape
    imgRec = np.zeros((nl,nc,3),dtype=np.uint8)
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    return imgRec 


# In[54]:


def add_padding(channel):
    """
    Adiciona padding a um canal de cor da imagem, copia a última linha e coluna
    até qie seja múltipla de 32.
    """
    linhas, colunas = channel.shape # dimensoes reais da imagem 
    
    # Calcula o novo número de linhas e colunas para ser múltiplo de 32
    linhasNovas = ((linhas - 1) // 32 + 1) * 32
    colunasNovas = ((colunas - 1) // 32 + 1) * 32
    
    # Cria uma nova matriz com as novas dimensões
    padded_channel = np.zeros((linhasNovas, colunasNovas), dtype=channel.dtype)
    
    # Copiar os valores originais da imagem
    padded_channel[:linhas, :colunas] = channel
    
    # Faz a cópia da última linha e coluna para o padding
    if linhasNovas > linhas:
        padded_channel[linhas:, :colunas] = channel[-1, :]
    if colunasNovas > colunas:
        padded_channel[:linhas, colunas:] = channel[:, -1][:, np.newaxis]
    if linhasNovas > linhas and colunasNovas > colunas:
        padded_channel[linhas:, colunas:] = channel[-1, -1]
    
    return padded_channel


# In[55]:


def remove_padding(padded_channel, original_shape):
    """
    Remove o padding de um canal de cor da imagem - passa p/ dimensão original.
    """
    original_nrows, original_ncols = original_shape
    return padded_channel[:original_nrows, :original_ncols]


# In[56]:


def rgb_to_ycbcr(R,G,B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B #Luminancia  
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128 # Crominancia Azul
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128 # Crominancia Vermelho
    return Y,Cb,Cr


# In[57]:


def ycbcr_to_rgb(Y,Cb,Cr):
    m = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    invm = np.linalg.inv(m)
    
    nl, nc = Y.shape
    
    # Cálculos com valores flutuantes
    R_float = invm[0][0] * Y + invm[0][1] * (Cb - 128) + invm[0][2] * (Cr - 128)
    G_float = invm[1][0] * Y + invm[1][1] * (Cb - 128) + invm[1][2] * (Cr - 128)
    B_float = invm[2][0] * Y + invm[2][1] * (Cb - 128) + invm[2][2] * (Cr - 128)
    
    R = np.round(R_float)
    G = np.round(G_float)
    B = np.round(B_float)
    
    R[R>255]=255
    G[G>255]=255
    B[B>255]=255
    
    R[R<0] = 0
    G[G<0] = 0
    B[B<0] = 0
    
    # Arredondar antes de converter para np.uint8
    R = R.astype(np.uint8)
    G = G.astype(np.uint8)
    B = B.astype(np.uint8)
    
    return R, G, B


# In[58]:


# Função para sub-amostrar os canais de uma imagem
def subamostrar(Y, Cb, Cr, fator_subamostra_horizontal, fator_subamostra_vertical, metodo='linear'):
    """
    Aplica downsampling aos canais Cb e Cr de uma imagem.
    """

    # Downsampling nos canais Y e Cb e Cr
    Y_d = Y.copy()
    Cb_d = cv2.resize(Cb, None, fx=fator_subamostra_horizontal, fy=fator_subamostra_vertical, interpolation=metodo)
    Cr_d = cv2.resize(Cr, None, fx=fator_subamostra_horizontal, fy=fator_subamostra_vertical, interpolation=metodo)
    
    return Y_d, Cb_d, Cr_d


# In[59]:


# Função para super-amostrar os canais de uma imagem
def superamostrar(Y_d, Cb_d, Cr_d, metodo='linear'):
    """
    Aplica upsampling aos canais Cb e Cr de uma imagem.
    """
    # Upsampling nos canais Cb e Cr
    Cb_up = cv2.resize(Cb_d, (Y_d.shape[1], Y_d.shape[0]), interpolation=metodo)
    Cr_up = cv2.resize(Cr_d, (Y_d.shape[1], Y_d.shape[0]), interpolation=metodo)
    
    return Y_d, Cb_up, Cr_up


# In[60]:


def encoder_dct(Y_d, Cb_d, Cr_d):
    """
    Aplica a DCT nos canais Y, Cb, e Cr sub-amostrados e visualiza os resultados.

    Args:
        Y_d, Cb_d, Cr_d (np.array): Canais Y, Cb, e Cr sub-amostrados.
    """
    # Aplica DCT aos canais
    Y_dct = calcular_dct_canal(Y_d)
    Cb_dct = calcular_dct_canal(Cb_d)
    Cr_dct = calcular_dct_canal(Cr_d)
    
    # Visualiza os canais DCT
    visualizar_canal_dct(Y_dct, "Y DCT")
    visualizar_canal_dct(Cb_dct, "Cb DCT")
    visualizar_canal_dct(Cr_dct, "Cr DCT")


# In[61]:


def calcular_dct_canal(canal):
    """
    Calcula a Transformada de Coseno Discreta (DCT) de um canal de imagem.

    Args:
        canal (np.array): Um canal de imagem (Y, Cb ou Cr) em formato de array NumPy.

    Returns:
        np.array: Canal transformado pela DCT.
    """
    # Aplica a DCT em 2D: primeiro ao longo das linhas, depois das colunas
    canal_dct = dct(dct(canal.T, norm='ortho').T, norm='ortho')
    
    return canal_dct


# In[62]:


def calcular_idct_canal(canal_dct):
    """
    Calcula a Transformada Inversa de Coseno Discreta (IDCT) de um canal de imagem transformado.

    Args:
        canal_dct (np.array): Um canal de imagem transformado pela DCT.

    Returns:
        np.array: Canal original recuperado pela IDCT.
    """
    # Aplica a IDCT em 2D: primeiro ao longo das linhas, depois das colunas
    canal_recuperado = idct(idct(canal_dct.T, norm='ortho').T, norm='ortho')
    return canal_recuperado
    
    


# In[63]:


def visualizar_canal_dct(canal_dct, titulo):
    """
    Visualiza um canal de imagem transformado pela DCT, usando transformação logarítmica.

    Args:
        canal_dct (np.array): Canal de imagem transformado pela DCT.
        titulo (str): Título para a janela da figura.
    """
    # Aplica a transformação logarítmica para melhor visualização
    imagem_log = np.log(np.abs(canal_dct) + 0.0001)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(imagem_log, cmap='gray')
    plt.title(titulo)
    plt.colorbar()
    plt.show()


# In[64]:


def aplicar_dct_blocos(canal, tamanho_bloco=8):
    altura, largura = canal.shape
    canal_dct = np.zeros_like(canal, dtype=float)
    
    for i in range(0, altura, tamanho_bloco):
        for j in range(0, largura, tamanho_bloco):
            bloco = canal[i:i+tamanho_bloco, j:j+tamanho_bloco]
           #bloco_padded = np.pad(bloco, ((0, tamanho_bloco - bloco.shape[0]), (0, tamanho_bloco - bloco.shape[1])), 'constant', constant_values=0)
            dct_bloco = dct(dct(bloco.T, norm='ortho').T, norm='ortho')
            
            # Corrigindo a inserção do bloco
            canal_dct[i:i+min(tamanho_bloco, bloco.shape[0]), j:j+min(tamanho_bloco, bloco.shape[1])] = dct_bloco[:bloco.shape[0], :bloco.shape[1]]
    
    return canal_dct


# In[65]:


def aplicar_idct_blocos(canal_dct, tamanho_bloco=8):
    """
    Aplica IDCT em blocos de tamanho especificado para recuperar um canal de imagem.

    Args:
        canal_dct (np.array): Canal de imagem transformado com DCT em blocos.
        tamanho_bloco (int): Tamanho do bloco, por exemplo, 8 ou 64.

    Returns:
        np.array: Canal de imagem recuperado com IDCT aplicada em blocos.
    """
    altura, largura = canal_dct.shape
    canal_recuperado = np.zeros_like(canal_dct)
    
    # Itera sobre blocos
    for i in range(0, altura, tamanho_bloco):
        for j in range(0, largura, tamanho_bloco):
            dct_bloco = canal_dct[i:i+tamanho_bloco, j:j+tamanho_bloco]
            idct_bloco = idct(idct(dct_bloco.T, norm='ortho').T, norm='ortho')
            canal_recuperado[i:i+tamanho_bloco, j:j+tamanho_bloco] = idct_bloco
    
    return canal_recuperado


# In[40]:


def quantizar_blocos_dct(canal_dct, matriz_quantizacao,qualidade):
    """
    Quantiza os coeficientes DCT de um canal de imagem, bloco a bloco.

    Args:
        canal_dct (np.array): Canal de imagem transformado pela DCT.
        matriz_quantizacao (np.array): Matriz de quantização específica para o canal (Y, Cb ou Cr).

    Returns:
        np.array: Canal de imagem com coeficientes DCT quantizados.
    """
    altura, largura = canal_dct.shape
    canal_quantizado = np.zeros_like(canal_dct)
    matriz_quantizacao=ajustar_matriz_quantizacao(matriz_quantizacao, qualidade)

    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            bloco_dct = canal_dct[i:i+8, j:j+8]
            bloco_quantizado = np.round(bloco_dct / matriz_quantizacao)
            canal_quantizado[i:i+8, j:j+8] = bloco_quantizado

    return canal_quantizado.astype(np.int16)


# In[113]:


def desquantizar_blocos(canal_quantizado, matriz_quantizacao, qualidade):
    """
    Desquantiza os coeficientes DCT de um canal de imagem, bloco a bloco.

    Args:
        canal_quantizado (np.array): Canal de imagem quantizado.
        matriz_quantizacao (np.array): Matriz de quantização específica para o canal (Y, Cb ou Cr).
        qualidade (int): Parâmetro de qualidade da imagem (entre 1 e 100).

    Returns:
        np.array: Canal de imagem com coeficientes DCT desquantizados.
    """
    altura, largura = canal_quantizado.shape
    canal_dct_desquantizado = np.zeros_like(canal_quantizado)
    matriz_quantizacao = ajustar_matriz_quantizacao(matriz_quantizacao, qualidade)

    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            bloco_quantizado = canal_quantizado[i:i+8, j:j+8]
            bloco_dct_desquantizado = bloco_quantizado * matriz_quantizacao
            canal_dct_desquantizado[i:i+8, j:j+8] = bloco_dct_desquantizado

    return canal_dct_desquantizado


# In[114]:


def ajustar_matriz_quantizacao(matriz_base, qualidade):
    if qualidade < 50:
        fator = 50 / qualidade
    else:
        fator = (100-qualidade) / 50
    if qualidade == 0:
        return np.ones_like(matriz_base)   
    
    matriz_ajustada = np.round((matriz_base * fator))
    matriz_ajustada[matriz_ajustada < 1] = 1
    matriz_ajustada[matriz_ajustada > 255] = 255
    
    
    return matriz_ajustada.astype(np.uint8)

# In[115]:
def codificar_dpcm_dc_imagem(quantizada):
    """
    Codifica os coeficientes DC de cada bloco na imagem quantizada usando DPCM.

    Args:
        quantizada (np.array): Imagem quantizada.

    Returns:
        np.array: Imagem com os coeficientes DC codificados usando DPCM.
    """
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

# In[116]:
def descodificar_dpcm_dc_imagem(imagem_codificada_dpcm):
    """
    Decodifica os coeficientes DC de cada bloco na imagem codificada com DPCM.

    Args:
        imagem_codificada_dpcm (np.array): Imagem com os coeficientes DC codificados usando DPCM.

    Returns:
        np.array: Imagem com os coeficientes DC descodificados.
    """
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

def diferencas(Y,Yr):
    diff= np.abs(Y-Yr)
    showImg(diff,"Diferencas",cmap="gray")
    
def erros(im_original, im_recuperada,Y,Yr):
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
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Signal-to-Noise Ratio (SNR): {snr}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr}")
    print(f"Maximum Difference: {max_diff}")
    print(f"Average Difference: {avg_diff}")
    
# In[70]:


def main():
    # 3.1. Ler imagem
    img = plt.imread('airport.bmp')
    
    # 3.4 - Encoder: Converte para YCbCr e aplica padding aos canais RGB
    Y_DPCM,Cb_DPCM,Cr_DPCM,original_shape,Y = encoder(img)
    Yr,imgrec=decoder(Y_DPCM,Cb_DPCM,Cr_DPCM,original_shape)
    
    diferencas(Y,Yr)
    erros(img,imgrec,Y,Yr)

    
    
    
if __name__ == "__main__":
    main()


# In[ ]: