"""
Student Performance (Math) — Pipeline de Classificação
======================================================
Autor:  Murilo Moretto Simões
Base:   Student Performance — Math (UCI, 395 instâncias, 30 atributos mistos)

Executar a partir da raiz do repositório:
    python codigo/murilo/student_performance.py
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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
DIR_DADOS = BASE_DIR / 'base de dados' / 'student+performance'
DIR_RESULTADOS = BASE_DIR / 'resultados' / 'murilo'
DIR_FIGURAS = DIR_RESULTADOS / 'figuras'

# ============================================================
# 1. CARREGAMENTO
# ============================================================

def carregar_student_performance():
    """
    Carrega a base Student Performance (Math).
    Atributos: 30 variáveis demográficas/escolares + notas G1, G2, G3
    Classe original: G3 (nota final, 0 a 20)
    """
    caminho = DIR_DADOS / 'student-mat.csv'
    df = pd.read_csv(caminho, sep=';')

    print(f"\n[Student Performance] Carregado: {df.shape[0]} instâncias, "
          f"{df.shape[1]} atributos")
    print(f"[Student Performance] Distribuição de G3:")
    print(df['G3'].describe())
    return df


# ============================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================

def preprocessar_student(df):
    """
    Pré-processamento da base Student Performance:
    1. Discretização de G3 em classe binária (Fail/Pass)
    2. Remoção da coluna G3 original
    3. Label Encoding nos atributos categóricos
    4. StandardScaler nos atributos numéricos
    5. Separação features (X) e classe (y)

    Nota sobre G1 e G2
    ------------------
    G1 e G2 são mantidos como features. São notas do 1º e 2º período
    que ajudam a prever o resultado final. Em um cenário real de predição
    antecipada não os teríamos, mas para fins acadêmicos são incluídos.
    Caso queira removê-los, descomente a linha indicada abaixo.

    Retorna
    -------
    X : pd.DataFrame com features pré-processadas
    y : pd.Series com a classe discretizada (0=Fail, 1=Pass)
    """
    df_proc = df.copy()

    # Discretizar G3 em classes: 0-9 = Fail (0), 10-20 = Pass (1)
    df_proc['class'] = df_proc['G3'].apply(lambda x: 0 if x < 10 else 1)

    # Remover G3 original
    df_proc = df_proc.drop(['G3'], axis=1)

    # Opcional: remover G1 e G2 (descomente se quiser removê-los)
    # df_proc = df_proc.drop(['G1', 'G2'], axis=1)

    # Identificar colunas categóricas e numéricas
    cat_cols = df_proc.select_dtypes(include=['object']).columns.tolist()
    num_cols = df_proc.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'class' in num_cols:
        num_cols.remove('class')

    # Label Encoding para categóricas
    le = LabelEncoder()
    for col in cat_cols:
        df_proc[col] = le.fit_transform(df_proc[col])

    # StandardScaler para numéricas (média=0, desvio padrão=1)
    scaler = StandardScaler()
    df_proc[num_cols] = scaler.fit_transform(df_proc[num_cols])

    X = df_proc.drop('class', axis=1)
    y = df_proc['class']

    print(f"\n[Student] Pré-processado: X={X.shape}, y={y.shape}")
    print(f"[Student] Distribuição da classe discretizada (0=Fail, 1=Pass):")
    print(y.value_counts())
    return X, y


# ============================================================
# 3. EXPERIMENTOS
# ============================================================

def executar_experimentos_student(X, y):
    """
    Executa todos os experimentos na base Student Performance.

    Algoritmos testados:
      - Árvore de Decisão (2 parametrizações)
      - GaussianNB (2 parametrizações)
      - Random Forest (2 parametrizações)
    """
    print(f"\n{'='*60}")
    print("EXPERIMENTOS: Student Performance (Math)")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = dividir_treino_teste(
        X, y, 'Student Performance'
    )
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
        'Árvore de Decisão', 'Student Performance',
        'Gini, sem poda (max_depth=None)'
    ))
    matriz_confusao_plot(
        dt1, X_test, y_test,
        'Árvore de Decisão (Gini)', 'Student Performance',
        DIR_FIGURAS / 'matriz_confusao_dt_gini.png',
        rotulos=['Fail', 'Pass']
    )

    dt2 = DecisionTreeClassifier(
        criterion='entropy', max_depth=5, random_state=RANDOM_STATE
    )
    resultados.append(avaliar_algoritmo(
        dt2, X_train, X_test, y_train, y_test,
        'Árvore de Decisão', 'Student Performance',
        'Entropia, max_depth=5'
    ))
    matriz_confusao_plot(
        dt2, X_test, y_test,
        'Árvore de Decisão (Entropia)', 'Student Performance',
        DIR_FIGURAS / 'matriz_confusao_dt_entropia.png',
        rotulos=['Fail', 'Pass']
    )

    # ----------------------------------------------------------
    # Naive Bayes — GaussianNB (dados com atributos numéricos)
    # ----------------------------------------------------------
    print("\n--- Naive Bayes (GaussianNB) ---")

    nb1 = GaussianNB(var_smoothing=1e-9)
    resultados.append(avaliar_algoritmo(
        nb1, X_train, X_test, y_train, y_test,
        'Naive Bayes', 'Student Performance',
        'GaussianNB, var_smoothing=1e-9'
    ))
    matriz_confusao_plot(
        nb1, X_test, y_test,
        'Naive Bayes (var_smoothing=1e-9)', 'Student Performance',
        DIR_FIGURAS / 'matriz_confusao_nb_1e-9.png',
        rotulos=['Fail', 'Pass']
    )

    nb2 = GaussianNB(var_smoothing=1e-7)
    resultados.append(avaliar_algoritmo(
        nb2, X_train, X_test, y_train, y_test,
        'Naive Bayes', 'Student Performance',
        'GaussianNB, var_smoothing=1e-7'
    ))
    matriz_confusao_plot(
        nb2, X_test, y_test,
        'Naive Bayes (var_smoothing=1e-7)', 'Student Performance',
        DIR_FIGURAS / 'matriz_confusao_nb_1e-7.png',
        rotulos=['Fail', 'Pass']
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
        'Random Forest', 'Student Performance',
        'n_estimators=100, max_depth=None'
    ))
    matriz_confusao_plot(
        rf1, X_test, y_test,
        'Random Forest (100 árvores)', 'Student Performance',
        DIR_FIGURAS / 'matriz_confusao_rf_100.png',
        rotulos=['Fail', 'Pass']
    )

    rf2 = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    )
    resultados.append(avaliar_algoritmo(
        rf2, X_train, X_test, y_train, y_test,
        'Random Forest', 'Student Performance',
        'n_estimators=200, max_depth=10'
    ))
    matriz_confusao_plot(
        rf2, X_test, y_test,
        'Random Forest (200 árvores)', 'Student Performance',
        DIR_FIGURAS / 'matriz_confusao_rf_200.png',
        rotulos=['Fail', 'Pass']
    )

    return resultados


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TRABALHO FINAL — CIÊNCIA DE DADOS 1")
    print("Student Performance (Math) — Murilo Moretto Simões")
    print("=" * 60)

    # Criar diretórios de saída
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    # Carregar
    df = carregar_student_performance()

    # Explorar
    explorar_dados(df, 'Student Performance', coluna_classe='G3')
    grafico_distribuicao_classe(
        df['G3'], 'Student Performance (G3 original)',
        DIR_FIGURAS / 'distribuicao_g3_original.png'
    )

    # Pré-processar
    X, y = preprocessar_student(df)

    # Gráfico da classe discretizada
    grafico_distribuicao_classe(
        y, 'Student Performance (Classe Discretizada)',
        DIR_FIGURAS / 'distribuicao_classe.png'
    )

    # Experimentos
    resultados = executar_experimentos_student(X, y)

    # Salvar tabela
    salvar_tabela(resultados, DIR_RESULTADOS / 'tabela_resultados_student.csv')

    print(f"\n{'='*60}")
    print("RESULTADOS SALVOS EM: resultados/murilo/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
