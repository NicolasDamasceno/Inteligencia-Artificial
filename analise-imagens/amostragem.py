import numpy as np
import cv2
import os

def amostragem(img, n):
  amostra = [lin[::n] for lin in img[::n]]
  return np.array(amostra)

if __name__ == '__main__':
  filename = 'analise-imagens/imagens/cliov6.jpg'

  fator = [2, 4]
  for ft in fator:
    img = cv2.imread(filename, 0)
    amostra = amostragem(img, ft)

    name, extension = os.path.splitext(filename)
    new_filename = '{name}-amostragem-{ft}{ext}'.format(name=name, ft=ft, ext=extension)
    cv2.imwrite(new_filename, amostra)
