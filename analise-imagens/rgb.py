import cv2

def media_RGB(alvo_x, alvo_y, caminho):
    img = cv2.imread(caminho)

    if img is None:
        print('Erro ao ler a imagem')

    altura, largura, canais = img.shape

    for y in range(altura):
        for x in range(largura):

            if x == alvo_x and y == alvo_y:
                # A ordem é Blue, Green e Red
                b, g, r = img[y, x]
                print(f'Canais de Cores B: {b}, G: {g}, R: {r}')
                media = (int(b) + int(g) + int(r)) / 3
                print(f'Média de Cores: {media:.2f}')

                # Desenha um pequeno círculo onde o Pixel está
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        
    cv2.imshow("Clio",img)
    return media

path = '/home/nicolas/Área de trabalho/Inteligencia-Artificial/analise-imagens/imagens/godzilla.jpg'
media_RGB(100, 150, path)