import numpy as np
import cv2
import os


def quantizacao_uniforme(img, K):
  a = np.float32(img)
  bucket = 256 / K
  quantizado = (a / (256 / K))
  return np.uint8(quantizado) * bucket

if __name__ == '__main__':
  
  filename = 'analise-imagens/imagens/godzilla.jpg'
  cores = [2, 8]

  for cor in cores:
    img = cv2.imread(filename, 0)
    resultado = quantizacao_uniforme(img, cor)
    name, extension = os.path.splitext(filename)
    new_filename = '{name}-quantizado-{k}{ext}'.format(name=name, k=cor, ext=extension)
    cv2.imwrite(new_filename, resultado)
