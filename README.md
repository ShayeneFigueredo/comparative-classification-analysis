# comparative-classification-analysis

**Trabalho Final — Ciência de Dados 1**  
UFU — Universidade Federal de Uberlândia

Comparação de desempenho de três algoritmos de classificação (Árvore de Decisão, Naive Bayes e Random Forest) aplicados a duas bases de dados do UCI Machine Learning Repository, seguindo metodologia padronizada de treino/teste e métricas de avaliação.

---

## 👥 Autores

| Nome | Base de dados | Código |
|------|--------------|--------|
| Shayene Lopes Figueredo | Car Evaluation (1728 instâncias) | `codigo/shayene/` |
| Murilo Moretto Simões | Student Performance — Math (395 instâncias) | `codigo/murilo/` |

---

## 🎯 Objetivo

Avaliar e comparar o desempenho de três classificadores em duas bases com naturezas distintas — uma **totalmente categórica** (Car Evaluation) e outra **mista** (Student Performance) — utilizando a mesma estratégia de divisão treino/teste, as mesmas parametrizações e as mesmas medidas de avaliação.

---

## 🧠 Algoritmos

| Algoritmo | Visto em sala? | Parametrizações testadas |
|-----------|:---:|--------------------------|
| Árvore de Decisão | ✅ | `gini` sem poda / `entropy` com `max_depth=5` |
| Naive Bayes | ✅ | `CategoricalNB` (alpha 1.0 / 0.5) ou `GaussianNB` |
| Random Forest | ❌ | 100 árvores sem poda / 200 árvores com `max_depth=10` |

---

## 📊 Metodologia

- **Divisão treino/teste:** Holdout 70/30 com estratificação (`random_state=42`)
- **Métricas:** Acurácia, F1-Score (weighted), Precision (weighted), Recall (weighted), Matriz de Confusão
- **Ferramenta:** Python 3 com scikit-learn, pandas, matplotlib e seaborn

---

## 📁 Estrutura do repositório

```
.
├── README.md
├── codigo/
│   ├── comum/
│   │   └── utilitarios.py              # Funções compartilhadas (métricas, gráficos)
│   ├── shayene/
│   │   └── car_evaluation.py           # Pipeline — Car Evaluation
│   └── murilo/
│       └── student_performance.py      # Pipeline — Student Performance
├── base de dados/
│   ├── car+evaluation/                 # Base Car Evaluation (UCI)
│   └── student+performance/            # Base Student Performance — Math (UCI)
└── resultados/
    ├── shayene/                        # Tabelas e gráficos da Shayene
    │   └── figuras/
    └── murilo/                         # Tabelas e gráficos do Murilo
        └── figuras/
```

---

## ▶️ Como executar

**Shayene (Car Evaluation):**
```bash
python codigo/shayene/car_evaluation.py
```

**Murilo (Student Performance):**
```bash
python codigo/murilo/student_performance.py
```

Os resultados (tabelas CSV e matrizes de confusão) serão gerados em `resultados/shayene/` e `resultados/murilo/` respectivamente.

---

## 📚 Fontes das bases de dados

Ambas as bases são do [UCI Machine Learning Repository](https://archive.ics.uci.edu/):

- **Car Evaluation:** [https://archive.ics.uci.edu/dataset/19/car+evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)
- **Student Performance:** [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
