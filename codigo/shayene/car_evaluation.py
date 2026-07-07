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

df = pd.read_csv('base_dados/car+evaluation/car.data', names=COLUNAS)

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

# print('Treino:', X_train.shape, '| Teste:', X_test.shape)


#config 1: treino do primeiro alg
from sklearn.tree import DecisionTreeClassifier

# criar e treinar a arvore
modelo = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
# print('Acuracia:', acc)

#config 2: com poda (max_depth=5)
modelo2 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
modelo2.fit(X_train, y_train)

y_pred2 = modelo2.predict(X_test)

acc2 = accuracy_score(y_test, y_pred2)
# print('Acuracia (com poda):', acc2)


# calculando F1-Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# funcao que avalia o modelo e devolve todas as metricas
def avaliar(modelo, X_test, y_test, nome, params):
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f'{nome} [{params}]')
    print(f'  Acurácia:  {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1-Score:  {f1:.4f}')
    print()
    return {'Algoritmo': nome, 'Params': params,
            'Acurácia': acc, 'F1': f1, 'Precision': prec, 'Recall': rec}

#avaliar os dois modelos que ja treinamos
# avaliar(modelo, X_test, y_test, 'Arvore de Decisao', 'Gini, sem poda')
# avaliar(modelo2, X_test, y_test, 'Árvore de Decisão', 'Entropy, max_depth=5')




# calculando naive bayes
from sklearn.naive_bayes import CategoricalNB

# config 1: alpha 1.0 (padrao)
nb1 = CategoricalNB(alpha=1.0)
nb1.fit(X_train, y_train)
# avaliar(nb1, X_test, y_test, 'Naive Bayes', 'alpha=1.0')

# conif 2: alpha = 0.5
nb2 = CategoricalNB(alpha=0.5)
nb2.fit(X_train, y_train)
# avaliar(nb2, X_test, y_test, 'Naive Bayes', 'alpha=0.5')



# calculando random forest
from sklearn.ensemble import RandomForestClassifier

#config 1: 100 arvores sem poda
rf1 = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf1.fit(X_train, y_train)
# avaliar(rf1, X_test, y_test, 'Random Forest', '100 arvores sem poda')

# config 2: 200 arvores com poda
rf2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf2.fit(X_train, y_train)
# avaliar(rf2, X_test, y_test, 'Random Forest', '200 arvores com poda')



# ============================================================
# AVALIAÇÃO FINAL (métricas + matrizes + tabela)
# ============================================================
from sklearn.metrics import confusion_matrix

# Lista com todos os modelos
modelos = [
    ('Árvore de Decisão', 'Gini, sem poda', modelo, 'matriz_dt_gini'),
    ('Árvore de Decisão', 'Entropy, max_depth=5', modelo2, 'matriz_dt_entropy'),
    ('Naive Bayes', 'alpha=1.0', nb1, 'matriz_nb_1.0'),
    ('Naive Bayes', 'alpha=0.5', nb2, 'matriz_nb_0.5'),
    ('Random Forest', '100 árvores, sem poda', rf1, 'matriz_rf_100'),
    ('Random Forest', '200 árvores, max_depth=10', rf2, 'matriz_rf_200'),
]

resultados = []

for nome, params, mod, arquivo in modelos:
    # Métricas
    res = avaliar(mod, X_test, y_test, nome, params)
    resultados.append(res)

    # Matriz de confusão
    y_pred = mod.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão — {nome}\n{params}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig(f'resultados/shayene/figuras/{arquivo}.png', dpi=150)
    print(f'[OK] {arquivo}.png')
    print()

# Salvar tabela CSV
tabela = pd.DataFrame(resultados)
tabela.to_csv('resultados/shayene/tabela_resultados.csv', index=False)
print('=' * 60)
print('TABELA FINAL DE RESULTADOS')
print('=' * 60)
print(tabela.to_string(index=False))
print()
print('Tabela salva em: resultados/shayene/tabela_resultados.csv')
print('Todas as 6 matrizes de confusão geradas em: resultados/shayene/figuras/')

