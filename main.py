import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as mcm
import numpy as np

#funções elementares:

# converter a imagem para RGB cinza
def grayscale(img):
  #esses coeficientes são uma convenção para converter para escala de cinza :)
  #e isso é necessário pois permite que o colormap pinte a imagem com base na intensidade de cinza
  return np.dot(img, [0.2989, 0.5870, 0.1140])

# exibir uma imagem em tons de cinza com um colormap aplicado e fornecer uma barra de cores como uma referência visual 
# para o mapeamento entre os valores de intensidade de cinza e as cores do colormap
def see_colormap(img, cm):
  #cria uma nova figura na qual a imagem será exibida.
  fig = plt.figure()
  #Exibir a imagem com o colormap
  plt.imshow(img, cm)
  #Adicionar uma barra de cores
  fig.colorbar(mcm.ScalarMappable(cmap=cm), ax=fig.gca())
  #Esconder os eixos
  plt.axis('off')
  #Mostrar a figura: exibir a figura inteira, que inclui a imagem com o colormap e a barra de cores correspondente, na tela
  plt.show()

# aplicar um mapa de cores personalizado a uma imagem em escala de cinza e depois visualizá-la com esse mapa de cores
def colormap(img, cmap):
  img = grayscale(img)
  #mapa de cores personalizado é criado a partir de uma lista de cores (cmap) fornecida pelo usuário.
  cm = clr.LinearSegmentedColormap.from_list('cmap', cmap, 256)
  #exibir a imagem em escala de cinza com o colormap aplicado
  see_colormap(img, cm)

#3.1. Leia uma imagem .bmp
def readImg(loc):
  return mpimg.imread(loc)

#3.2. Crie uma função para implementar um colormap definido pelo utilizador
def user_colormap(img):
 #arrumar 
 colormap(img, colors)

#3.4. Encoder: Crie uma função para separar a imagem nos seus componentes RGB. 
def encoder(img, isolate_rgb = False):
  if isolate_rgb:
    return isolate_rgb(img)
  
#3.5. Decoder: Crie também a função inversa (que combine os 3 componentes RGB).
def decoder(img, unite_rgb = False):
  if unite_rgb:
    return unite_rgb(img)
  
#3.3. Crie uma função que permita visualizar a imagem com um dado colormap.
nature = readImg('nature.bmp')
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

#show_rgb_channels(nature)




def pad_channel(channel):
  """
Função auxiliar para fazer o padding de um canal RGB numa imagem.
Recebe o canal que queremos fazer padding, o canal tem as mesmas dimensões que a imagem em si (mesmo número de linhas e colunas, aparentemente)
Retorna o canal com padding quando necessário, caso contrário retorna o canal original
"""

  n_linhas = channel.shape[0] #número de linhas
  n_colunas = channel.shape[1] #número de colunas

  

  #passo 1: calcular o resto da divisão por 32 para verificar se as dimensões são múltiplas de 32

  resto_linhas = n_linhas%32
  resto_colunas = n_colunas%32


  #passo 2: determinar o padding necessário

  if resto_linhas==0:
    pad_linhas = 0
  else:
    pad_linhas = 32-resto_linhas
    

  if resto_colunas==0:
    pad_colunas = 0
  else:
    pad_colunas = 32-resto_colunas


  # aplicar o padding a um canal quando necessário
  if pad_colunas > 0 or pad_linhas > 0:
      channel_padded = np.pad(channel, ((0, pad_linhas), (0, pad_colunas)), mode='edge')
      return channel_padded
  else:
      return channel
    




def pad_image(img_name):
    """
    Adiciona o padding necessário para tornar as dimensões das linhas e colunas dos canais da array da imagem múltiplos de 32
    Recebe o nome da imagem como argumento, por exemplo: "nature.bmp"
    Retorna uma imagem com o padding apropriado
    """
    
    
    image = readImg(img_name) #ler a imagem

    red, green, blue = isolate_rgb(image) #separar os canais RGB para receberem o padding
    red_padded = pad_channel(red)
    blue_padded = pad_channel(blue)
    green_padded = pad_channel(green)

    padded_image = np.stack((red_padded, green_padded, blue_padded), axis=-1) #voltar a juntar tudo
    """
    Nota: A imagem final é um array com 3 dimensões: linhas, colunas e canais.
    O axis = -1 serve para garantir o empilhamento final dos canais no eixo correto dos canais
    """
    
   

        

    return padded_image




       

#padded_image = pad_image("nature.bmp");

"""
Função que remove o padding da imagem
"""
def unpad_image(padded_image,original_dim):
    linhas_originais, colunas_originais = original_dim[:2]
    unpadded_img = padded_image[:linhas_originais, :colunas_originais, :]
    return unpadded_img








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

padded_img, og = padding(nature)

unpadded_img = unpadding(padded_img, og)

print(f'Dimensão Original: {nature.shape[:2]}')
print(f'Dimensão Padded: {padded_img.shape[:2]}\nDimensão Unpadded: {unpadded_img.shape[:2]}')


def RGB_to_YCbCr(img):
  cm = np.array([[0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312]])


def encoder(img, isolate_rgb = False):
  if isolate_rgb:
    return isolate_rgb(img)

def decoder(img, unite_rgb = False):
  if unite_rgb:
    return unite_rgb(img)

#show image
  

def main():
  nature = readImg('nature.bmp')
  user_colormap(nature)
  show_rgb_channels(nature)
  

if __name__ == "__main__":
    main()

