"""
Car Evaluation — Pipeline de Classificação
==========================================
Autora: Shayene Lopes Figueredo
Base:   Car Evaluation (UCI, 1728 instâncias, 6 atributos categóricos)

Executar a partir da raiz do repositório:
    python codigo/shayene/car_evaluation.py
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier

# Ajusta o path para importar do módulo comum
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comum.utilitarios import (
    RANDOM_STATE,
    explorar_dados,
    grafico_distribuicao_classe,
    dividir_treino_teste,
    avaliar_algoritmo,
    matriz_confusao_plot,
    salvar_tabela,
)

# ============================================================
# CAMINHOS (relativos à raiz do repositório)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # raiz do repo
DIR_DADOS = BASE_DIR / 'base de dados' / 'car+evaluation'
DIR_RESULTADOS = BASE_DIR / 'resultados' / 'shayene'
DIR_FIGURAS = DIR_RESULTADOS / 'figuras'

# ============================================================
# 1. CARREGAMENTO
# ============================================================

def carregar_car_evaluation():
    """
    Carrega a base Car Evaluation.
    Atributos: buying, maint, doors, persons, lug_boot, safety
    Classe: class (unacc, acc, good, vgood)
    """
    colunas = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    caminho = DIR_DADOS / 'car.data'
    df = pd.read_csv(caminho, names=colunas)

    print(f"\n[Car Evaluation] Carregado: {df.shape[0]} instâncias, "
          f"{df.shape[1]} atributos")
    print(f"[Car Evaluation] Distribuição da classe:")
    print(df['class'].value_counts())
    return df


# ============================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================

def preprocessar_car(df):
    """
    Pré-processamento da base Car Evaluation:
    - Label Encoding em TODAS as colunas (atributos + classe)
    - Separação features (X) e classe (y)

    Retorna
    -------
    X : pd.DataFrame com 6 colunas de features codificadas
    y : pd.Series com a classe codificada
    """
    df_encoded = df.copy()

    le = LabelEncoder()
    for col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Mostrar mapeamento da classe (para referência)
    mapeamento_classe = dict(zip(
        le.classes_,
        le.transform(le.classes_)
    ))
    print(f"\n[Car] Mapeamento da classe: {mapeamento_classe}")

    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']

    print(f"[Car] Pré-processado: X={X.shape}, y={y.shape}")
    return X, y


# ============================================================
# 3. EXPERIMENTOS
# ============================================================

def executar_experimentos_car(X, y):
    """
    Executa todos os experimentos na base Car Evaluation.

    Algoritmos testados:
      - Árvore de Decisão (2 parametrizações)
      - CategoricalNB (2 parametrizações)
      - Random Forest (2 parametrizações)
    """
    print(f"\n{'='*60}")
    print("EXPERIMENTOS: Car Evaluation")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y, 'Car Evaluation')
    resultados = []

    # ----------------------------------------------------------
    # Árvore de Decisão
    # ----------------------------------------------------------
    print("\n--- Árvore de Decisão ---")

    dt1 = DecisionTreeClassifier(
        criterion='gini', max_depth=None, random_state=RANDOM_STATE
    )
    resultados.append(avaliar_algoritmo(
        dt1, X_train, X_test, y_train, y_test,
        'Árvore de Decisão', 'Car Evaluation',
        'Gini, sem poda (max_depth=None)'
    ))
    matriz_confusao_plot(
        dt1, X_test, y_test,
        'Árvore de Decisão (Gini)', 'Car Evaluation',
        DIR_FIGURAS / 'matriz_confusao_dt_gini.png'
    )

    dt2 = DecisionTreeClassifier(
        criterion='entropy', max_depth=5, random_state=RANDOM_STATE
    )
    resultados.append(avaliar_algoritmo(
        dt2, X_train, X_test, y_train, y_test,
        'Árvore de Decisão', 'Car Evaluation',
        'Entropia, max_depth=5'
    ))
    matriz_confusao_plot(
        dt2, X_test, y_test,
        'Árvore de Decisão (Entropia)', 'Car Evaluation',
        DIR_FIGURAS / 'matriz_confusao_dt_entropia.png'
    )

    # ----------------------------------------------------------
    # Naive Bayes — CategoricalNB (dados 100% categóricos)
    # ----------------------------------------------------------
    print("\n--- Naive Bayes (CategoricalNB) ---")

    nb1 = CategoricalNB(alpha=1.0)
    resultados.append(avaliar_algoritmo(
        nb1, X_train, X_test, y_train, y_test,
        'Naive Bayes', 'Car Evaluation',
        'CategoricalNB, alpha=1.0'
    ))
    matriz_confusao_plot(
        nb1, X_test, y_test,
        'Naive Bayes (alpha=1.0)', 'Car Evaluation',
        DIR_FIGURAS / 'matriz_confusao_nb_alpha1.0.png'
    )

    nb2 = CategoricalNB(alpha=0.5)
    resultados.append(avaliar_algoritmo(
        nb2, X_train, X_test, y_train, y_test,
        'Naive Bayes', 'Car Evaluation',
        'CategoricalNB, alpha=0.5'
    ))
    matriz_confusao_plot(
        nb2, X_test, y_test,
        'Naive Bayes (alpha=0.5)', 'Car Evaluation',
        DIR_FIGURAS / 'matriz_confusao_nb_alpha0.5.png'
    )

    # ----------------------------------------------------------
    # Random Forest
    # ----------------------------------------------------------
    print("\n--- Random Forest ---")

    rf1 = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=RANDOM_STATE
    )
    resultados.append(avaliar_algoritmo(
        rf1, X_train, X_test, y_train, y_test,
        'Random Forest', 'Car Evaluation',
        'n_estimators=100, max_depth=None'
    ))
    matriz_confusao_plot(
        rf1, X_test, y_test,
        'Random Forest (100 árvores)', 'Car Evaluation',
        DIR_FIGURAS / 'matriz_confusao_rf_100.png'
    )

    rf2 = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    )
    resultados.append(avaliar_algoritmo(
        rf2, X_train, X_test, y_train, y_test,
        'Random Forest', 'Car Evaluation',
        'n_estimators=200, max_depth=10'
    ))
    matriz_confusao_plot(
        rf2, X_test, y_test,
        'Random Forest (200 árvores)', 'Car Evaluation',
        DIR_FIGURAS / 'matriz_confusao_rf_200.png'
    )

    return resultados


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TRABALHO FINAL — CIÊNCIA DE DADOS 1")
    print("Car Evaluation — Shayene Lopes Figueredo")
    print("=" * 60)

    # Criar diretórios de saída
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    # Carregar
    df = carregar_car_evaluation()

    # Explorar
    explorar_dados(df, 'Car Evaluation', coluna_classe='class')
    grafico_distribuicao_classe(
        df['class'], 'Car Evaluation',
        DIR_FIGURAS / 'distribuicao_classe.png'
    )

    # Pré-processar
    X, y = preprocessar_car(df)

    # Experimentos
    resultados = executar_experimentos_car(X, y)

    # Salvar tabela
    salvar_tabela(resultados, DIR_RESULTADOS / 'tabela_resultados_car.csv')

    print(f"\n{'='*60}")
    print("RESULTADOS SALVOS EM: resultados/shayene/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
