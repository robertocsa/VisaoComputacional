from collections import deque, defaultdict
import ast

# Mapeamento de cores
cores = {
    'Vermelho': 0,
    'Laranja': 1,
    'Azul': 2,
    'Verde escuro': 3,
    'Amarelo': 4,
    'Roxo': 5
}
cores_inv = {v: k for k, v in cores.items()}

# Pesos das heurísticas
ALFA = 1.0   # Peso para células de fronteira
BETA = 0.8   # Peso para ocorrências totais da cor no tabuleiro
GAMMA = 5.0  # Peso para expansão simulada

def ler_tabuleiro(nome_arquivo):
    matriz = []
    with open(nome_arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            linha_limpa = linha.strip()
            if ':' in linha_limpa:
                linha_limpa = linha_limpa.split(':', 1)[1].strip()
            lista_de_cores = ast.literal_eval(linha_limpa)
            linha_codificada = [cores[cor] for cor in lista_de_cores]
            matriz.append(linha_codificada)
    return matriz

def flood_fill_bfs(matriz, nova_cor):
    n, m = len(matriz), len(matriz[0])
    cor_original = matriz[0][0]
    if cor_original == nova_cor:
        return
    fila = deque([(0, 0)])
    visitado = [[False]*m for _ in range(n)]
    visitado[0][0] = True
    while fila:
        x, y = fila.popleft()
        matriz[x][y] = nova_cor
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < n and 0 <= ny < m and not visitado[nx][ny] and matriz[nx][ny] == cor_original:
                visitado[nx][ny] = True
                fila.append((nx, ny))

def obter_regiao(matriz, cor):
    n, m = len(matriz), len(matriz[0])
    regiao = set()
    fila = deque([(0, 0)])
    visitado = [[False]*m for _ in range(n)]
    visitado[0][0] = True
    while fila:
        x, y = fila.popleft()
        if matriz[x][y] == cor:
            regiao.add((x, y))
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < n and 0 <= ny < m and not visitado[nx][ny] and matriz[nx][ny] == cor:
                    visitado[nx][ny] = True
                    fila.append((nx, ny))
    return regiao

def contar_fronteiras(matriz, regiao, cor_atual):
    n, m = len(matriz), len(matriz[0])
    fronteiras = defaultdict(int)
    for x, y in regiao:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in regiao:
                cor = matriz[nx][ny]
                if cor != cor_atual:
                    fronteiras[cor] += 1
    return fronteiras

def contar_ocorrencias_totais(matriz):
    contagem = defaultdict(int)
    for linha in matriz:
        for cor in linha:
            contagem[cor] += 1
    return contagem

def escolher_melhor_cor(tabuleiro, regiao, cor_atual):
    fronteiras = contar_fronteiras(tabuleiro, regiao, cor_atual)
    totais = contar_ocorrencias_totais(tabuleiro)
    melhor_cor = cor_atual
    melhor_pontuacao = -1

    print("[DEBUG] Avaliando pontuações por cor:")
    for cor in fronteiras:
        if cor == cor_atual:
            continue

        copia = [linha[:] for linha in tabuleiro]
        flood_fill_bfs(copia, cor)
        nova_regiao = obter_regiao(copia, cor)
        tamanho_expansao = len(nova_regiao)

        fronteira = fronteiras[cor]
        total = totais[cor]
        pontuacao = ALFA * fronteira + BETA * total + GAMMA * tamanho_expansao

        print(f"  - {cores_inv[cor]}: fronteira={fronteira}, total={total}, expansão={tamanho_expansao}, pontuação={pontuacao:.2f}")

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_cor = cor

    return melhor_cor

def jogar(nome_arquivo):
    tabuleiro = ler_tabuleiro(nome_arquivo)
    for i in range(31):
        print(f"\n[DEBUG] Jogada {i+1}")
        cor_atual = tabuleiro[0][0]
        print(f"[DEBUG] Cor atual na origem: {cores_inv[cor_atual]}")
        regiao = obter_regiao(tabuleiro, cor_atual)
        print(f"[DEBUG] Tamanho da região atual: {len(regiao)}")
        fronteiras = contar_fronteiras(tabuleiro, regiao, cor_atual)
        print(f"[DEBUG] Fronteiras detectadas:")
        for cor, qtd in fronteiras.items():
            print(f"  - {cores_inv[cor]}: {qtd} células adjacentes")
        proxima_cor = escolher_melhor_cor(tabuleiro, regiao, cor_atual)
        print(f"[DEBUG] Cor escolhida: {cores_inv[proxima_cor]}")
        flood_fill_bfs(tabuleiro, proxima_cor)
        print(f"[DEBUG] Flood fill: de {cores_inv[cor_atual]} para {cores_inv[proxima_cor]}")
        if all(cell == tabuleiro[0][0] for row in tabuleiro for cell in row):
            print(f"\n[SUCESSO] Tabuleiro unificado com a cor {cores_inv[tabuleiro[0][0]]} em {i+1} jogadas.")
            return
    print("\n[FIM] Não foi possível unificar o tabuleiro em 31 jogadas.")

# Executa o jogo
if __name__ == "__main__":
    jogar("entrada.txt")
