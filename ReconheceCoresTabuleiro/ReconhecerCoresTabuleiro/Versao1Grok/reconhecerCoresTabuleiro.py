# Criado na maior parte com o Grok (https://grok.com/chat/5e666c4b-62aa-467f-b01d-63c661fdc0d7)

import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def identify_dominant_color(region, tolerance=80, debug_cell=None):
    """Identifica a cor exata em uma região da imagem usando K-means."""
    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    
    if debug_cell:
        print(f"\nDebug - Célula {debug_cell}: Valores RGB originais (antes de qualquer processamento):")
        print(region_rgb)
        print(f"  - Forma da região: {region_rgb.shape}")
        print(f"  - Média direta dos pixels (antes de K-means): {np.mean(region_rgb, axis=(0, 1)).astype(int)}")
    
    pixels = region_rgb.reshape(-1, 3)
    if debug_cell:
        print(f"  - Pixels achatados: {pixels.shape[0]} pixels")
        print(f"  - Valores únicos dos pixels: {np.unique(pixels, axis=0)}")
    
    kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    if debug_cell:
        print(f"  - Cor dominante após K-means: {dominant_color}")
        print(f"  - Centroide bruto (antes de arredondar): {kmeans.cluster_centers_[0]}")
    
    colors = {
        (255, 0, 0): "Vermelho",
        (0, 128, 0): "Verde escuro",
        (0, 0, 255): "Azul",
        (255, 165, 0): "Laranja",
        (255, 255, 0): "Amarelo",
        (128, 0, 128): "Roxo"
    }
    
    min_dist = float('inf')
    closest_color = None
    for color_rgb, color_name in colors.items():
        dist = np.linalg.norm(dominant_color - np.array(color_rgb))
        if debug_cell:
            print(f"  - Distância para {color_name} ({color_rgb}): {dist:.2f}")
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    
    if closest_color in ["Amarelo", "Laranja"]:
        if debug_cell:
            print(f"  - Ajuste fino para Amarelo/Laranja iniciado...")
        if min_dist > 50:
            if dominant_color[1] < 200:
                closest_color = "Laranja"
                if debug_cell:
                    print(f"  - Decisão: Verde < 200, classificado como Laranja")
            else:
                closest_color = "Amarelo"
                if debug_cell:
                    print(f"  - Decisão: Verde >= 200, classificado como Amarelo")
    
    if min_dist > tolerance:
        if debug_cell:
            print(f"Aviso: Cor dominante {dominant_color} fora da tolerância ({min_dist:.2f}), usando cor mais próxima: {closest_color}")
    
    return closest_color, dominant_color

def process_board_image(image_path, grid_size=(17, 17)):
    """Processa a imagem do tabuleiro 17x17 e retorna a matriz de cores."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Não foi possível carregar a imagem.")
    
    # Confirmar dimensões da imagem
    height, width = image.shape[:2]
    print(f"Debug - Dimensões da imagem: {width}x{height} pixels")
    
    adjusted_image = image
    
    gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(script_dir, "binary_image.png"), thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Nenhum contorno encontrado na imagem.")
    
    contour_image = image.copy()
    board_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(board_contour)
    print(f"Debug - Coordenadas do contorno do tabuleiro: (x={x}, y={y}, w={w}, h={h})")
    
    # Ajuste manual das coordenadas com base no GIMP
    board_start_x, board_start_y = 3, 3  # Início do tabuleiro ajustado
    cell_width, cell_height = 35, 35  # Tamanho fixo com base no GIMP (595/17 ≈ 35)
    margin = 2  # Margem interna de 2 pixels para evitar bordas
    
    color_matrix = []
    
    print("Valores RGB dominantes e cores identificadas por célula:")
    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            start_x = board_start_x + j * cell_width
            start_y = board_start_y + i * cell_height
            end_x = start_x + cell_width
            end_y = start_y + cell_height
            
            # Aplicar margem interna
            start_x += margin
            start_y += margin
            end_x -= margin
            end_y -= margin
            
            cell = adjusted_image[start_y:end_y, start_x:end_x]
            
            if i + 1 == 14 and j + 1 == 14:
                print(f"Debug - Coordenadas ajustadas da célula (14, 14): (start_x={start_x}, start_y={start_y}, end_x={end_x}, end_y={end_y})")
            
            if cell.size == 0:
                row.append("Desconhecida")
                print(f"Célula ({i+1}, {j+1}): Desconhecida (célula vazia)")
                continue
            
            debug_cell = (i+1, j+1) if (i+1 == 14 and j+1 == 14) else None
            color, dominant_rgb = identify_dominant_color(cell, debug_cell=debug_cell)
            print(f"Célula ({i+1}, {j+1}): RGB dominante = {dominant_rgb}, Cor identificada = {color}")
            row.append(color)
        color_matrix.append(row)
    
    return color_matrix

def main():
    if len(sys.argv) < 2:
        print("Uso: python script.py <caminho_da_imagem>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    grid_size = (17, 17)
    
    try:
        color_matrix = process_board_image(image_path, grid_size)
        print("\nMatriz de cores identificadas (17x17):")
        for i, row in enumerate(color_matrix):
            print((i+1), row)
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    import sys
    main()