import cv2  # biblioteca de processamento de imagens
import numpy as np


def redimensiona_imagens(imagens: list, tamanho: tuple) -> list:
    """
    Redimensiona as imagens.

    :param imagens: Tupla de imagens.
    :param tamanho: Tamanho (largura, altura) para o redimensionamento.
    :return: Imagens redimensionadas.
    """
    redimensionadas = []
    for img in imagens:
        redimensionadas.append(cv2.resize(img, tamanho, interpolation = cv2.INTER_LINEAR))
    return redimensionadas


def morphing(imagem, alphas, tamanho):
    """
    Gera uma interpolação afim para a imagem e alpha passados.

    :param imagem: Uma imagem.
    :param alphas: Os alphas calculados.
    :return: A imagem interpolada.
    """
    pixels = []

    lab_images = [cv2.cvtColor(img, cv2.COLOR_BGR2Lab) for img in imagem]  # converte as imagem de BGR para CIELAB
    # transforma a imagem em vetores de pixels    
    for img in lab_images:
        pixels.append(img.reshape((-1, 3)))

    # interpolação afim - processo de morphing
    morphed_pixels = np.zeros_like(pixels[0], dtype=np.float64)
    for i in range(len(imagem)):
        morphed_pixels += alphas[i] * pixels[i]

    # converte os vetores de pixels de volta para o formato de imagem
    morphed_image = morphed_pixels.reshape((tamanho[1], tamanho[0], 3))

    # converte a imagem de volta para o espaço de cores BGR
    morphed_image = cv2.cvtColor(np.uint8(morphed_image), cv2.COLOR_Lab2BGR)

    return morphed_image


def main(imagens, tamanho, n_morphing):
    """
    :param imagens: Imagens a serem utilizadas no processo de morphing.
    :param tamanho: Tamanho para o qual as imagens serão redimensionadas.
    :param n_morphing: Número de etapas do morphing.
    """
    # redimensiona todas as imagens para o mesmo tamanho
    imagens = redimensiona_imagens(imagens, tamanho)

    # número de interpolações entre cada par de imagens
    n_interpolacoes = n_morphing // (len(imagens) - 1)

    while True:
        for i in range(len(imagens) - 1):
            for n in range(n_interpolacoes + 1):
                # cálcula o alpha
                alpha = n / n_interpolacoes
                # gera o n_morphing
                morphed = morphing([imagens[i], imagens[i + 1]], [1 - alpha, alpha], tamanho)

                # exibe a imagem
                cv2.imshow('Imagem Interpolada', morphed)

                # verifica se o esc foi pressionado
                key = cv2.waitKey(1)
                if key == 27:
                    break
            if key == 27:
                break
        if key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    imagens = []  # array de imagens
    N_MORPHING = 300 # número de etapas do morphing.

    imagens.append(cv2.imread('./img/mona1.jpg'))  # mona
    imagens.append(cv2.imread('./img/mona2.jpg'))  # mona

    main(imagens, [450, 650], N_MORPHING)  # roda o programa
