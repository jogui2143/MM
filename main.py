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

user_colormap(nature)

def encoder():

  return "encoder"

def decoder():
  # Some code here
  return "decoder"

#show image


