import skimage.util
import skimage.io


"""
Modes:
  gaussian
  localvar
  poisson
  salt
  pepper
  s&p
  speckle
  None
"""
def plot_noise(img, mode="gaussian"):
  if mode is None:
    return img
  else: #chapuza
    img = skimage.io.imread(img)/255.0
    array = skimage.util.random_noise(img, mode=mode)
    matplotlib.pyplot.imsave('/content/tmp.png', array)
    return '/content/tmp.png'
    