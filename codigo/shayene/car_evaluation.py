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

# print('Features:', X.shape)
# print('Classe:', y.shape)


#treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print('Treino:', X_train.shape, '| Teste:', X_test.shape)


#config 1: treino do primeiro alg
from sklearn.tree import DecisionTreeClassifier

# criar e treinar a arvore
modelo = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
modelo.fit(X_train, y_train)

# fazer previsoes no teste
y_pred = modelo.predict(X_test)

# medir acuracia
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('Acuracia:', acc)



#config 2: com poda (max_depth=5)
modelo2 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
modelo2.fit(X_train, y_train)
y_pred2 = modelo2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
print('Acuracia (com poda):', acc2)