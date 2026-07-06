"""
Car Evaluation — Pipeline de Classificação
==========================================
Autora: Shayene Lopes Figueredo
Base:   Car Evaluation (UCI, 1728 instâncias, 6 atributos categóricos)

Executar a partir da raiz do repositório:
    python codigo/shayene/car_evaluation.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dei nome para as colunas pq o arquivo nao tem cabeçalho
COLUNAS = ['p_compra', 'p_manutencao', 'portas', 'capacidade', 'porta_malas', 'seguranca', 'classe']

df= pd.read_csv('base_dados/car+evaluation/car.data', names=COLUNAS)

#ver quantos carros sao unacc, acc, good e vgood
# print(df['classe'].value_counts())

#criando grafico de barras da distribuição de classe
df['classe'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Distribuição da Classe - Car Evaluation")
plt.xlabel('Classe')
plt.ylabel('Quantidade')
# plt.savefig('resultados/shayene/figuras/distribuicao_classe.png', dpi=150)


#PROCESSAMENTO DOS DADOS
from sklearn.preprocessing import LabelEncoder

#criando copia do dataframe pra nao bagunçar o original
df_cod = df.copy()

le = LabelEncoder()

for coluna in df_cod.columns:
    df_cod[coluna] = le.fit_transform(df_cod[coluna])

# print(df_cod.head())
# print()
# print(df_cod['classe'].value_counts())


#separer features (x) da classe (y)
X = df_cod.drop('classe', axis=1)
y = df_cod['classe']

print('Features:', X.shape)
print('Classe:', y.shape)
