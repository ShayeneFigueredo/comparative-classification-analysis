"""
Análise da base Student Performance, disciplina matemática.
Autor: Murilo Moretto Simões
Disciplina: Ciência de Dados 1 — Trabalho Final
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ───────────────────────────────────────────────────────
# Constantes (combinadas entre os dois — NÃO ALTERAR)
# ───────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Diretórios de saída
DIR_RESULTADOS = "resultados/murilo"
DIR_FIGURAS = os.path.join(DIR_RESULTADOS, "figuras")


def carregar_dados():
    """
    Carrega a base Student Performance (Matemática).
    """
    df = pd.read_csv("base_dados/student+performance/student-mat.csv", sep=";")

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
    Cria a variável alvo de classificação a partir de G3.

    0 = Reprovado (G3 < 10)
    1 = Aprovado  (G3 >= 10)
    """
    df["class"] = (df["G3"] >= 10).astype(int)

    # Remove a coluna G3 original, substituída pela classe
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


def grafico_classes(df):
    """
    Gera e SALVA um gráfico de barras da distribuição das classes.
    """
    os.makedirs(DIR_FIGURAS, exist_ok=True)

    distribuicao = df["class"].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Reprovado (0)", "Aprovado (1)"],
        [distribuicao[0], distribuicao[1]],
        color=["tomato", "steelblue"],
        edgecolor="black"
    )
    plt.title("Distribuição de Classes — Student Performance (Math)")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de alunos")

    caminho = os.path.join(DIR_FIGURAS, "distribuicao_classe.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"[OK] Gráfico salvo: {caminho}")


def codificar_e_normalizar(df):
    """
    1. Label Encoding nas colunas categóricas (texto → número).
    2. StandardScaler nas colunas numéricas (média=0, desvio padrão=1).

    O StandardScaler é necessário para o GaussianNB, que assume
    que os atributos seguem uma distribuição normal. Atributos em
    escalas diferentes (ex: idade 15-22 vs. faltas 0-75) precisam
    ser equalizados para que o cálculo de variância seja justo.

    IMPORTANTE: As colunas numéricas são identificadas ANTES do
    Label Encoding, para que as categóricas recém-codificadas
    não sejam normalizadas indevidamente.
    """

    # --- PASSO 0: Identificar colunas ANTES da codificação ---
    colunas_categoricas = df.select_dtypes(include=["object"]).columns.tolist()

    # Colunas numéricas = int64/float64, excluindo 'class' (alvo)
    colunas_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    colunas_numericas = [c for c in colunas_numericas if c != "class"]

    print(f"\nColunas categóricas → Label Encoding: {len(colunas_categoricas)}")
    print(colunas_categoricas)
    print(f"\nColunas numéricas → StandardScaler: {len(colunas_numericas)}")
    print(colunas_numericas)

    # --- PASSO 1: Label Encoding nas colunas categóricas ---
    encoder = LabelEncoder()
    for coluna in colunas_categoricas:
        df[coluna] = encoder.fit_transform(df[coluna])

    # --- PASSO 2: StandardScaler SOMENTE nas numéricas originais ---
    scaler = StandardScaler()
    df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

    print("\n===== TIPOS APÓS PRÉ-PROCESSAMENTO =====")
    df.info()

    return df


def separar_treino_teste(df):
    """
    Separa a base em treino (70%) e teste (30%) com estratificação.
    """
    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n===== DIVISÃO TREINO/TESTE =====")
    print(f"Treino: {X_train.shape[0]} registros ({X_train.shape[0]/df.shape[0]*100:.0f}%)")
    print(f"Teste:  {X_test.shape[0]} registros ({X_test.shape[0]/df.shape[0]*100:.0f}%)")
    print(f"Features (X): {X_train.shape[1]} atributos")
    print(f"Classe (y): 2 classes (0=Reprovado, 1=Aprovado)")

    return X_train, X_test, y_train, y_test


# ───────────────────────────────────────────────────────
# Função de avaliação PADRÃO (métricas + matriz + tabela)
# ───────────────────────────────────────────────────────

def avaliar_e_salvar(modelo, X_test, y_test, nome, params, arquivo_matriz):
    """
    Avalia um modelo treinado com as 4 métricas (todas weighted),
    gera a matriz de confusão em PNG e retorna um dicionário
    com os resultados para a tabela final.
    """
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n===== {nome} — {params} =====")
    print(f"  Acurácia:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(cm)

    # Salvar matriz como PNG
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Reprovado (0)", "Aprovado (1)"],
        yticklabels=["Reprovado (0)", "Aprovado (1)"]
    )
    plt.title(f"Matriz de Confusão — {nome}\n{params}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    caminho = os.path.join(DIR_FIGURAS, f"{arquivo_matriz}.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"  [OK] Matriz salva: {caminho}")

    return {
        "Algoritmo": nome,
        "Parametrização": params,
        "Acurácia": round(acc, 4),
        "F1-Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4)
    }


# ───────────────────────────────────────────────────────
# Funções de treino de cada modelo
# ───────────────────────────────────────────────────────

def arvore_decisao_gini(X_train, X_test, y_train, y_test):
    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)
    return avaliar_e_salvar(
        modelo, X_test, y_test,
        "Árvore de Decisão", "Gini, sem poda",
        "matriz_dt_gini"
    )


def arvore_decisao_entropy(X_train, X_test, y_train, y_test):
    modelo = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)
    return avaliar_e_salvar(
        modelo, X_test, y_test,
        "Árvore de Decisão", "Entropy, max_depth=5",
        "matriz_dt_entropy"
    )


def naive_bayes_default(X_train, X_test, y_train, y_test):
    modelo = GaussianNB(var_smoothing=1e-9)
    modelo.fit(X_train, y_train)
    return avaliar_e_salvar(
        modelo, X_test, y_test,
        "Naive Bayes", "GaussianNB, var_smoothing=1e-9",
        "matriz_nb_1e-9"
    )


def naive_bayes_smoothing(X_train, X_test, y_train, y_test):
    modelo = GaussianNB(var_smoothing=1e-7)
    modelo.fit(X_train, y_train)
    return avaliar_e_salvar(
        modelo, X_test, y_test,
        "Naive Bayes", "GaussianNB, var_smoothing=1e-7",
        "matriz_nb_1e-7"
    )


def random_forest_default(X_train, X_test, y_train, y_test):
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)
    return avaliar_e_salvar(
        modelo, X_test, y_test,
        "Random Forest", "100 árvores, sem poda",
        "matriz_rf_100"
    )


def random_forest_poda(X_train, X_test, y_train, y_test):
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)
    return avaliar_e_salvar(
        modelo, X_test, y_test,
        "Random Forest", "200 árvores, max_depth=10",
        "matriz_rf_200"
    )


# ───────────────────────────────────────────────────────
# Pipeline principal
# ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STUDENT PERFORMANCE (MATH) — PIPELINE DE CLASSIFICAÇÃO")
    print("Autor: Murilo Moretto Simões")
    print("=" * 60)

    # 1. Carregar
    df = carregar_dados()

    # 2. Explorar
    explorar_dados(df)

    # 3. Criar classe
    df = criar_variavel_classe(df)
    analisar_classes(df)

    # 4. Gráfico de distribuição
    grafico_classes(df)

    # 5. Pré-processamento: Label Encoding + StandardScaler
    df = codificar_e_normalizar(df)

    # 6. Separar treino e teste
    X_train, X_test, y_train, y_test = separar_treino_teste(df)

    # 7. Rodar os 6 experimentos
    print("\n" + "=" * 60)
    print("EXECUTANDO OS 6 EXPERIMENTOS")
    print("=" * 60)

    resultados = []

    resultados.append(
        arvore_decisao_gini(X_train, X_test, y_train, y_test)
    )
    resultados.append(
        arvore_decisao_entropy(X_train, X_test, y_train, y_test)
    )
    resultados.append(
        naive_bayes_default(X_train, X_test, y_train, y_test)
    )
    resultados.append(
        naive_bayes_smoothing(X_train, X_test, y_train, y_test)
    )
    resultados.append(
        random_forest_default(X_train, X_test, y_train, y_test)
    )
    resultados.append(
        random_forest_poda(X_train, X_test, y_train, y_test)
    )

    # 8. Salvar tabela CSV
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    tabela = pd.DataFrame(resultados)
    caminho_csv = os.path.join(DIR_RESULTADOS, "tabela_resultados.csv")
    tabela.to_csv(caminho_csv, index=False)

    print("\n" + "=" * 60)
    print("TABELA FINAL DE RESULTADOS")
    print("=" * 60)
    print(tabela.to_string(index=False))
    print(f"\nTabela salva em: {caminho_csv}")
    print(f"Matrizes de confusão salvas em: {DIR_FIGURAS}/")
    print("=" * 60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    main()
