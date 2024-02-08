import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as mcm
import numpy as np

def readImg(loc):
  return mpimg.imread(loc)

nature = readImg('nature.bmp')

def grayscale(img):
  #esses coeficientes são uma convenção para converter para escala de cinza :)
  #e isso é necessário pois permite que o colormap pinte a imagem com base na intensidade de cinza
  return np.dot(img, [0.2989, 0.5870, 0.1140])

def colormap(img, cmap):
  img = grayscale(img)
  cm = clr.LinearSegmentedColormap.from_list('cmap', cmap, 256)
  see_colormap(img, cm)

def see_colormap(img, cm):
  fig = plt.figure()
  plt.imshow(img, cm)
  fig.colorbar(mcm.ScalarMappable(cmap=cm), ax=fig.gca())
  plt.axis('off')
  plt.show()

def user_colormap(img):
 colors = input('Insert colors: ').split(' ')
 colors += ['black'] if len(colors) < 2 else colors
 colormap(img, colors)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### Teste do colormap
#user_colormap(nature)

def isolate_rgb(img):
  return img[:,:,0], img[:,:,1], img[:,:,2]

def unite_rgb(r, g, b):
  return np.dstack((r, g, b))

def show_imgs(files, images, cm=None):
  if cm == None:
    cm = list()
  #if len(cm) < len(images):
   # cm += ['Greys'] * (len(images) - len(cm))
  fig = plt.figure()
  for i in range(len(images)):
    fig.add_subplot(len(images)-1//3 +1, ((len(images)-1)%3) +1, i+1)
    plt.title(files[i])
    plt.imshow(images[i], cmap= cm[i])
    plt.axis('off')
  plt.show()

def show_rgb_channels(img):
  r, g, b = isolate_rgb(img)
  red = clr.LinearSegmentedColormap.from_list('cmap', ['black', 'red'], N=256)
  green = clr.LinearSegmentedColormap.from_list('cmap', ['black', 'green'], N=256)
  blue = clr.LinearSegmentedColormap.from_list('cmap', ['black', 'blue'], N=256)
  show_imgs(['Red', 'Green', 'Blue'], (r, g, b), cm=[red, green, blue])

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### Teste do isolamento e união dos canais RGB
#show_rgb_channels(nature)

def padding(img):
  # Captura a altura (h) e a largura (ln) da imagem.
  h,w = img.shape[:2]
  p1, p2, p3 = isolate_rgb(img)
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

def unpadding(img, og):
  return img[:og[0], :og[1], :]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### Teste do padding e unpadding
#padded_img, og = padding(nature)

#unpadded_img = unpadding(padded_img, og)

#print(f'Dimensão Original: {nature.shape[:2]}')
#print(f'Dimensão Padded: {padded_img.shape[:2]}\nDimensão Unpadded: {unpadded_img.shape[:2]}')


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

  return np.dstack((y, cb, cr))

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

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### Teste do RGB para YCbCr e YCbCr para RGB
#nature_YcbCr = RGB_to_YCbCr(nature)
#nature_RGB = YCbCr_to_RGB(nature_YcbCr)
#print(f'Pixel 0x0 original: {nature_RGB[0, 0, :]}')
#print(f'Pixel 0x0 dps: {nature[0, 0, :]}')
#print(f'Pixel 0x0 equivalente: {nature[0, 0, :] == nature_RGB[0, 0, :]}')

def encoder(img, isolate_rgb = False, padding = False, rgb_to_ycbcr = False):
  if isolate_rgb:
    return isolate_rgb(img)
  elif padding:
    return padding(img)
  elif rgb_to_ycbcr:
    return RGB_to_YCbCr(img)

def decoder(img, unite_rgb = False, unpadding = False, ycbcr_to_rgb = False):
  if unite_rgb:
    return unite_rgb(img)
  elif unpadding:
    return unpadding(img)
  elif ycbcr_to_rgb:
    return YCbCr_to_RGB(img)


