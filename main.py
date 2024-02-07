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

show_rgb_channels(nature)

def encoder(img, isolate_rgb = False):
  if isolate_rgb:
    return isolate_rgb(img)

def decoder(img, unite_rgb = False):
  if unite_rgb:
    return unite_rgb(img)

#show image


