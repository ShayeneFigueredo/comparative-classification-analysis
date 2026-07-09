"""
Análise da base Student Performance, disciplina matemática.
Autor: Murilo Moretto Simões
Disciplina: Ciência de Dados
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def carregar_dados():
    """
    Carrega a base Student Performance (Matemática).
    """
    df = pd.read_csv("student-mat.csv", sep=";")

    print("Base carregada com sucesso!")
    print(f"Número de linhas: {df.shape[0]}")
    print(f"Número de colunas: {df.shape[1]}")
    return df


def explorar_dados(df):
    """Realiza a exploração inicial da base."""

    print("\n===== PRIMEIRAS LINHAS =====")
    print(df.head())

    print("\n===== INFORMAÇÕES DA BASE =====")
    df.info()

    print("\n===== ESTATÍSTICAS DESCRITIVAS =====")
    print(df.describe())

    print("\n===== VALORES AUSENTES =====")
    print(df.isnull().sum())


def criar_variavel_classe(df):
    """
    Cria a variável alvo de classificação.

    0 = Reprovado
    1 = Aprovado
    """

    df["class"] = (df["G3"] >= 10).astype(int)

    df.drop(columns=["G3"], inplace=True)

    return df


def analisar_classes(df):
    """
    Exibe a distribuição das classes da variável alvo.
    """

    print("\n===== DISTRIBUIÇÃO DAS CLASSES =====")
    print(df["class"].value_counts())

    print("\n===== DISTRIBUIÇÃO EM PORCENTAGEM =====")
    print(df["class"].value_counts(normalize=True) * 100)


def codificar_atributos(df):
    """
    Converte todos os atributos categóricos em valores numéricos.
    """

    encoder = LabelEncoder()

    colunas_categoricas = df.select_dtypes(include=["object"]).columns

    for coluna in colunas_categoricas:
        df[coluna] = encoder.fit_transform(df[coluna])

    'Verifica se ainda existe alguma coluna object, resposta deve ser não. Todos devem ser int64'
    print("\n===== TIPOS APÓS LABEL ENCODER =====")
    df.info()

    return df

    "Removewmos apenas a coluna Classe, todo restante continua em X"
    X = df.drop(columns=["class"])
    y = df["class"]


def separar_treino_teste(df):
    """
    Separa a base em treino e teste.
    """

    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("\n===== DIVISÃO DA BASE =====")
    print(f"Treino: {X_train.shape[0]} registros")
    print(f"Teste: {X_test.shape[0]} registros")

    return X_train, X_test, y_train, y_test


def grafico_classes(df):
    """
    Gera um gráfico de barras da distribuição das classes.
    """

    distribuicao = df["class"].value_counts()

    plt.figure(figsize=(6, 4))

    plt.bar(["Aprovado", "Reprovado"],
            [distribuicao[1], distribuicao[0]])

    plt.title("Distribuição das Classes")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de alunos")

    plt.show()


def arvore_decisao_gini(X_train, X_test, y_train, y_test):

    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n===== ÁRVORE DE DECISÃO (GINI) =====")

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Precisão:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    matriz = confusion_matrix(y_test, y_pred)

    print("\nMatriz de Confusão:")
    print(matriz)


def arvore_decisao_entropy(X_train, X_test, y_train, y_test):

    modelo = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n===== ÁRVORE DE DECISÃO (ENTROPY) =====")

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Precisão:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    matriz = confusion_matrix(y_test, y_pred)

    print("\nMatriz de Confusão:")
    print(matriz)


def naive_bayes_default(X_train, X_test, y_train, y_test):

    modelo = GaussianNB(
        var_smoothing=1e-9
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n===== GAUSSIAN NAIVE BAYES (DEFAULT) =====")

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Precisão:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    matriz = confusion_matrix(y_test, y_pred)

    print("\nMatriz de Confusão:")
    print(matriz)


def naive_bayes_smoothing(X_train, X_test, y_train, y_test):

    modelo = GaussianNB(
        var_smoothing=1e-7
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n===== GAUSSIAN NAIVE BAYES (SMOOTHING) =====")

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Precisão:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    matriz = confusion_matrix(y_test, y_pred)

    print("\nMatriz de Confusão:")
    print(matriz)


def random_forest_default(X_train, X_test, y_train, y_test):

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n===== RANDOM FOREST (100 ÁRVORES) =====")

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Precisão:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    matriz = confusion_matrix(y_test, y_pred)

    print("\nMatriz de Confusão:")
    print(matriz)


def random_forest_poda(X_train, X_test, y_train, y_test):

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n===== RANDOM FOREST (200 ÁRVORES / PODA) =====")

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Precisão:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    matriz = confusion_matrix(y_test, y_pred)

    print("\nMatriz de Confusão:")
    print(matriz)


def main():
    df = carregar_dados()

    explorar_dados(df)

    df = criar_variavel_classe(df)

    analisar_classes(df)

    grafico_classes(df)

    df = codificar_atributos(df)

    X_train, X_test, y_train, y_test = separar_treino_teste(df)

    arvore_decisao_gini(
        X_train,
        X_test,
        y_train,
        y_test
    )

    arvore_decisao_entropy(
        X_train,
        X_test,
        y_train,
        y_test
    )

    naive_bayes_default(
        X_train,
        X_test,
        y_train,
        y_test
    )

    naive_bayes_smoothing(
        X_train,
        X_test,
        y_train,
        y_test
    )

    random_forest_default(
        X_train,
        X_test,
        y_train,
        y_test
    )

    random_forest_poda(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()
