"""
Utilitários compartilhados — Trabalho Final CD1
===============================================
Funções de avaliação, métricas e visualização usadas por ambas as bases.

Importe deste módulo para garantir que a comparação entre Car Evaluation
e Student Performance use exatamente as mesmas funções de métrica.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ============================================================
# CONFIGURAÇÕES GLOBAIS (NÃO ALTERAR — combinadas entre os dois)
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.3  # Holdout 70/30


# ============================================================
# EXPLORAÇÃO DE DADOS
# ============================================================

def explorar_dados(df, nome_base, coluna_classe):
    """
    Gera informações exploratórias sobre a base.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    nome_base : str
        Nome da base para exibição.
    coluna_classe : str
        Nome da coluna que contém a classe.
    """
    print(f"\n{'='*60}")
    print(f"EXPLORAÇÃO: {nome_base}")
    print(f"{'='*60}")

    print(f"\nShape: {df.shape}")
    print(f"\nTipos dos dados:\n{df.dtypes}")
    print(f"\nValores ausentes:\n{df.isnull().sum()}")
    print(f"\nEstatísticas descritivas (numéricos):\n{df.describe()}")

    print(f"\nDistribuição da classe '{coluna_classe}':")
    print(df[coluna_classe].value_counts())
    print(f"\nProporções:\n{df[coluna_classe].value_counts(normalize=True)}")


def grafico_distribuicao_classe(y, nome_base, caminho_saida):
    """
    Gera e salva gráfico de barras da distribuição da classe.

    Parâmetros
    ----------
    y : pd.Series
        Série com os valores da classe.
    nome_base : str
        Nome da base para o título.
    caminho_saida : str
        Caminho completo onde salvar o PNG.
    """
    plt.figure(figsize=(6, 4))
    y.value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(f'Distribuição da Classe - {nome_base}')
    plt.xlabel('Classe')
    plt.ylabel('Nº de Instâncias')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close()
    print(f"[OK] Gráfico salvo: {caminho_saida}")


# ============================================================
# DIVISÃO TREINO/TESTE
# ============================================================

def dividir_treino_teste(X, y, nome_base):
    """
    Divide os dados em treino (70%) e teste (30%) com estratificação.

    Parâmetros
    ----------
    X : pd.DataFrame ou np.ndarray
        Features.
    y : pd.Series ou np.ndarray
        Classe.
    nome_base : str
        Nome da base para exibição.

    Retorna
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[{nome_base}] Treino: {X_train.shape[0]} instâncias | "
          f"Teste: {X_test.shape[0]} instâncias")
    return X_train, X_test, y_train, y_test


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar_algoritmo(modelo, X_train, X_test, y_train, y_test,
                      nome_modelo, nome_base, parametrizacao=""):
    """
    Treina o modelo e retorna as métricas de avaliação.

    Parâmetros
    ----------
    modelo : estimator
        Modelo scikit-learn já configurado.
    X_train, X_test, y_train, y_test : arrays
        Dados de treino e teste.
    nome_modelo : str
        Nome do algoritmo (ex: 'Árvore de Decisão').
    nome_base : str
        Nome da base (ex: 'Car Evaluation').
    parametrizacao : str
        Descrição da parametrização usada.

    Retorna
    -------
    dict com as métricas calculadas.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    resultados = {
        'Algoritmo': nome_modelo,
        'Base': nome_base,
        'Parametrização': parametrizacao,
        'Acurácia': round(acc, 4),
        'Precision (weighted)': round(prec, 4),
        'Recall (weighted)': round(rec, 4),
        'F1-Score (weighted)': round(f1, 4),
    }

    print(f"  {nome_modelo} [{parametrizacao}] → "
          f"Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    return resultados


def avaliar_cross_validation(modelo, X, y, nome_modelo, parametrizacao, cv=5):
    """
    Avalia o modelo usando validação cruzada estratificada.

    Parâmetros
    ----------
    modelo : estimator
        Modelo scikit-learn já configurado.
    X, y : arrays
        Dados completos.
    nome_modelo : str
        Nome do algoritmo.
    parametrizacao : str
        Descrição da parametrização.
    cv : int
        Número de folds (default: 5).

    Retorna
    -------
    dict com médias e desvios das métricas.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = cross_val_score(modelo, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(modelo, X, y, cv=skf, scoring='f1_weighted')

    print(f"  CV-{cv} {nome_modelo} [{parametrizacao}] → "
          f"Acc média: {acc_scores.mean():.4f} (±{acc_scores.std():.4f}) | "
          f"F1 média: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")

    return {
        'Algoritmo': nome_modelo,
        'Parametrização': parametrizacao,
        'Acurácia (CV mean)': round(acc_scores.mean(), 4),
        'Acurácia (CV std)': round(acc_scores.std(), 4),
        'F1 (CV mean)': round(f1_scores.mean(), 4),
        'F1 (CV std)': round(f1_scores.std(), 4),
    }


def matriz_confusao_plot(modelo, X_test, y_test, nome_modelo,
                         nome_base, caminho_saida, rotulos=None):
    """
    Gera e salva a matriz de confusão.

    Parâmetros
    ----------
    modelo : estimator
        Modelo já treinado.
    X_test, y_test : arrays
        Dados de teste.
    nome_modelo : str
        Nome do algoritmo.
    nome_base : str
        Nome da base.
    caminho_saida : str
        Caminho completo onde salvar o PNG.
    rotulos : list, opcional
        Rótulos para os eixos da matriz.
    """
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=rotulos, yticklabels=rotulos)
    plt.title(f'Matriz de Confusão - {nome_modelo}\n{nome_base}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close()
    print(f"[OK] Matriz de confusão salva: {caminho_saida}")


def salvar_tabela(resultados, caminho_saida):
    """
    Salva a lista de resultados como CSV.

    Parâmetros
    ----------
    resultados : list[dict]
        Lista de dicionários com as métricas.
    caminho_saida : str
        Caminho completo onde salvar o CSV.
    """
    df = pd.DataFrame(resultados)
    df.to_csv(caminho_saida, index=False)
    print(f"\n[OK] Tabela salva: {caminho_saida}")
    print(df.to_string(index=False))
    return df
