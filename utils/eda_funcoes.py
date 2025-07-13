import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Função para verificar os dados que são outliers
def valores_outliers(base, coluna):
    q1 = np.percentile(base[coluna], 25)
    q3 = np.percentile(base[coluna], 75)
    iqr = q3 - q1
    
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    outliers = base[coluna].loc[(base[coluna] < limite_inferior) | (base[coluna] > limite_superior)]
    
    return outliers, (limite_inferior, limite_superior)


# Função para mostrar a distribuição de dados numéricos
def mostrar_distribuicao_dados_numericos(base, coluna):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=base[coluna], kde=True)
    plt.title(f"Distribuição de {coluna}")
    plt.subplot(1, 2, 2)
    sns.boxplot(data=base, y=coluna)
    plt.title(f"Boxplot de {coluna}")
    plt.tight_layout()
    plt.show()
    
# Função para mostrar a relação de uma variável numérica e outra categórica através de um boxplot
def gerar_boxplot(base, x, y):
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=base, y=y, x=x)
    plt.title(f"Boxplot em Relação a {y} e se {x}")
    plt.xticks(ticks=[0, 1], labels=['Não', 'Sim']);
    
    
# Função para calcular a regra de Sturges para dividir os números em classes
def regra_sturges(base, coluna):
    # tamanho
    n = base[coluna].shape[0]
    #regra de sturges
    k = 1 + 3.3 * np.log10(n)
    k = int(k.round(0))
    return k


# Função para transformar variável numérica em uma variável categórica através de classes
def segmentar_dados_numericos_categoricos(base, coluna):
    # Criando uma cópia do dataframe original
    base2 = base.copy()
    
    # Usando a regra de sturges para criar as classes
    classes = regra_sturges(base2, coluna)
    
    # Segmentando os dados numéricos em dados categóricos
    base2[coluna + '_classe'] = pd.cut(base2[coluna], bins=classes, right=False)
    
    return base2


# Função para gerar uma tabela de contigência
def gerar_tabela_contigencia(base, linha, coluna):
    cross_tab = pd.crosstab(base[linha], base[coluna], normalize='columns')
    cross_tab.columns = ['Bom', 'Ruim']
    
    return cross_tab

# Função para calcular o IV (Information Value)
def calcular_iv(base, linha, coluna_alvo):
    
    # Gerar uma tabela de contingência
    cross_tab = gerar_tabela_contigencia(base, linha, coluna_alvo)
    
    # Calcular o WoE
    cross_tab['WoE'] = np.log(cross_tab['Bom'] / cross_tab['Ruim'])
    
    # Calcular o IV
    cross_tab['IV'] = (cross_tab['Bom'] - cross_tab['Ruim']) * cross_tab['WoE']
    
    # Apagando as linhas em que os valores apareceram inf por ter ocorrido uma divisão por zero
    cross_tab = cross_tab.replace([np.inf, -np.inf], np.nan).dropna(subset=['IV'])
    
    iv_value = cross_tab['IV'].sum()
    
    if iv_value < 0.02:
        print(f"IV de {linha}: {iv_value:.2f}\nPoder de separação: Muito fraco")
    elif iv_value < 0.1:
        print(f"IV de {linha}: {iv_value:.2f}\nPoder de separação: Fraco")
    elif iv_value < 0.3:
        print(f"IV de {linha}: {iv_value:.2f}\nPoder de separação: Médio")
    elif iv_value <= 0.5:
        print(f"IV de {linha}: {iv_value:.2f}\nPoder de separação: Forte")
    else:
        print(f"IV de {linha}: {iv_value:.2f}\nPoder de separação: Muito bom para ser verdade")

