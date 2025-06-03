"""
Analisador de Cores Dominantes em Tabuleiro 17x17 com OpenCV e KMeans

Este script analisa uma imagem de um tabuleiro (por exemplo, de um jogo) com 17x17 células,
identificando a cor dominante em cada célula e retornando uma matriz com os nomes das cores.

Imagem de tabuleiro fonte de inspiração:
https://www.advanced-ict.info/interactive/flood_fill.html

========================
REQUISITOS:
------------------------
- Python 3.x
- Bibliotecas:
    - opencv-python (cv2)
    - numpy
    - scikit-learn

Para instalar as dependências:
    pip install opencv-python numpy scikit-learn

========================
USO:
------------------------
    python script.py caminho/para/imagem.png

Certifique-se de que:
- A imagem tenha um tabuleiro 17x17, com tamanho aproximado de 595x595 px
- A posição e tamanho das células sejam ajustados conforme a imagem analisada
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import sys

def identify_dominant_color(region, tolerance=80, debug_cell=None):
    """
    Identifica a cor dominante em uma região da imagem usando K-means.

    Parâmetros:
        region (np.array): região da imagem (célula) em BGR
        tolerance (int): tolerância para ajuste fino da cor
        debug_cell (tuple): (opcional) coordenadas da célula para debug

    Retorno:
        Tuple[str, Tuple[int, int, int]]: nome da cor e RGB dominante
    """
    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    pixels = region_rgb.reshape(-1, 3)

    # Aplica K-means para encontrar a cor dominante
    kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)

    # Dicionário de cores base
    colors = {
        (255, 0, 0): "Vermelho",
        (0, 128, 0): "Verde escuro",
        (0, 0, 255): "Azul",
        (255, 165, 0): "Laranja",
        (255, 255, 0): "Amarelo",
        (128, 0, 128): "Roxo"
    }

    # Encontra a cor mais próxima no dicionário
    min_dist = float('inf')
    closest_color = None
    for color_rgb, color_name in colors.items():
        dist = np.linalg.norm(dominant_color - np.array(color_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name

    # Ajuste fino para amarelo e laranja
    if closest_color in ["Amarelo", "Laranja"] and min_dist > 50:
        if dominant_color[1] < 200:
            closest_color = "Laranja"
        else:
            closest_color = "Amarelo"

    # Caso a cor esteja fora da tolerância
    if min_dist > tolerance:
        pass  # Ainda assim retornamos a mais próxima

    return closest_color, dominant_color

def process_board_image(image_path, grid_size=(17, 17)):
    """
    Processa uma imagem de tabuleiro 17x17 e identifica a cor dominante em cada célula.

    Parâmetros:
        image_path (str): caminho da imagem
        grid_size (tuple): dimensão da grade do tabuleiro (default 17x17)

    Retorno:
        List[List[str]]: matriz com nomes das cores detectadas por célula
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Não foi possível carregar a imagem.")

    height, width = image.shape[:2]

    # Pré-processamento com limiar adaptativo
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite(os.path.join(script_dir, "binary_image.png"), thresh)

    # Encontra o maior contorno (presumivelmente o tabuleiro)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Nenhum contorno encontrado na imagem.")

    board_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(board_contour)

    # Parâmetros ajustados manualmente com base na imagem de referência
    board_start_x, board_start_y = 3, 3
    cell_width, cell_height = 35, 35    #Considera as dimensões totais da imagem (595x595) divididas por 17, ou seja: 595/17=35
    margin = 2

    color_matrix = []

    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            # Calcula coordenadas da célula
            start_x = board_start_x + j * cell_width + margin
            start_y = board_start_y + i * cell_height + margin
            end_x = start_x + cell_width - 2 * margin
            end_y = start_y + cell_height - 2 * margin

            cell = image[start_y:end_y, start_x:end_x]

            if cell.size == 0:
                row.append("Desconhecida")
                continue

            color, _ = identify_dominant_color(cell)
            row.append(color)
        color_matrix.append(row)

    return color_matrix

def main():
    """
    Função principal: carrega a imagem e imprime a matriz de cores do tabuleiro.
    """
    if len(sys.argv) < 2:
        print("Uso: python script.py <caminho_da_imagem>")
        sys.exit(1)

    image_path = sys.argv[1]
    grid_size = (17, 17)

    try:
        color_matrix = process_board_image(image_path, grid_size)
        print("\nMatriz de cores identificadas (17x17):")
        for i, row in enumerate(color_matrix):
            print(f"Linha {i+1}: {row}")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
