# Previsão de Churn de Clientes - Análise de Caso Olist

<p align="center">
  <img src="assets/banner_churn.png" width="700">
</p>

---

📌 ## Descrição do Projeto

Este repositório contém o desenvolvimento de uma solução de *Data Science* para a Olist, focada na previsão de cancelamento de clientes (churn). O objetivo principal é construir um modelo de machine learning que identifique clientes com alta probabilidade de churn, permitindo a criação de campanhas de retenção proativas para minimizar perdas de receita e aumentar a fidelidade do cliente.

📄 **Veja a apresentação do projeto:** [Apresentacao-Resultados-de-Churn.pdf](report/Apresentacao-Resultados-de-Churn.pdf)
<br>
📄 **Veja a análise exploratória:** [analise_exploratoria.ipynb](notebook/analise_exploratoria.ipynb)
<br>
📄 **Veja a modelagem e os resultados:** [processamento_modelagem.ipynb.ipynb](notebook/processamento_modelagem.ipynb)

---

💼 ## Entendimento do Negócio

No competitivo mercado de e-commerce, o custo de aquisição de um novo cliente é significativamente maior do que o custo de reter um cliente existente. O churn (cancelamento) representa uma perda direta de receita e um indicador da saúde do negócio e da satisfação do cliente.

Este projeto utiliza dados transacionais e de comportamento para desenvolver um modelo de classificação capaz de prever quais clientes estão em risco de churn. Com essa informação, a Olist pode direcionar ações de marketing e relacionamento, otimizando investimentos e fortalecendo sua base de clientes.

#### Principais KPIs (Indicadores Chave de Desempenho) de Retenção:

* **Revocação (Recall):** Percentual de clientes que realmente dariam churn e que foram corretamente identificados pelo modelo. Essencial para garantir que a campanha de retenção atinja o máximo de clientes em risco.
* **Valor Líquido Gerado:** Impacto financeiro final da solução, calculado pela receita salva menos os custos da campanha de retenção e as perdas remanescentes.
* **LTV (Lifetime Value) Salvo:** Valor total de receita futura que foi preservada ao reter os clientes identificados.
* **Custo de Retenção:** Custo associado às ações de marketing (descontos, ofertas) para reter um cliente.

---

🛠️ ## Pré-Processamento dos Dados

Para garantir a qualidade e a relevância das informações, foram realizadas as seguintes etapas de pré-processamento:

* **Limpeza e Tratamento:** Remoção de dados inconsistentes e tratamento de outliers.
* **Seleção de Variáveis:** Foram removidas variáveis que não agregam valor preditivo ou que representam vazamento de dados (data leakage), como:
    * Identificadores únicos (`order_id`, `customer_id`, etc.).
* **Pipeline de Transformação:** Foi criado um `Pipeline` do Scikit-learn para automatizar o pré-processamento, garantindo consistência e reprodutibilidade. O pipeline incluiu:
    1.  **Tratamento de Variáveis Categóricas:** Utilização de encoders para transformar categorias em formato numérico.
    2.  **Tratamento de Missing Values:** Preenchimento de valores faltantes com a mediana para variáveis numéricas e com a moda para variáveis categóricas.

---

🤖 ## Modelagem

O problema de churn é caracterizado por um desbalanceamento de classes, onde o número de clientes que dão churn é muito menor que o de clientes ativos. Essa característica foi considerada durante todo o processo de modelagem.

* **Validação do Modelo:** A separação entre treino e teste foi feita de forma **estratificada por cliente**, garantindo que todas as transações de um mesmo cliente estivessem apenas em um dos conjuntos, evitando data leakage.
* **Modelos Testados:** Para encontrar a melhor solução, foram avaliados quatro algoritmos de classificação:
    * Regressão Logística
    * LightGBM
    * **XGBoost (Modelo Vencedor)**
    * CatBoost

#### Análise de Importância de Features com SHAP

Após a tunagem de hiperparâmetros, o **XGBoost** foi o modelo com melhor performance. Para entender quais fatores mais influenciam suas decisões, utilizamos a análise **SHAP (SHapley Additive exPlanations)**.

![SHAP Summary Plot](assets/importancia_global.png)

A análise revelou que o **`valor do frete`** é, de longe, o fator com maior impacto nas previsões de churn, seguido pelo preço e número de parcelas. Isso indica que a experiência de entrega é um ponto crítico para a satisfação e retenção dos clientes da Olist.

#### Tunagem de Hiperparâmetros e Resultados Finais

O modelo XGBoost passou por um processo de tunagem de hiperparâmetros com o objetivo de maximizar a métrica **ROC AUC**. Os resultados finais no conjunto de teste foram:

* **ROC AUC:** 0.74
* **Revocação (Recall):** 0.72
* **Precisão:** 0.36
* **Acurácia:** 0.64
* **Medida F1:** 0.48

A alta revocação indica que o modelo é capaz de identificar 72% de todos os clientes que realmente iriam cancelar, o que é fundamental para o sucesso das campanhas de retenção.

![Resultados Finais do Modelo](assets/resultados_finais_modelo.png)

---

💡 ## Desempenho Financeiro e Estratégia

O verdadeiro valor de um modelo preditivo está em seu impacto no negócio. Simulamos a implementação do modelo em um cenário real para quantificar seu retorno financeiro.

* **Cenário de Referência (Sem Modelo):** A empresa enfrentaria uma perda estimada de **R$ 744,131.33**.
* **Cenário Otimizado (Com Modelo):** Ao direcionar uma campanha de retenção com base nas previsões:
    * **Receita Salva:** R$ 166,852.87
    * **Custo da Campanha:** R$ 192,360.49
    * **Perda Remanescente:** R$ 187,955.10

### Conclusão Estratégica
A implementação do modelo preditivo de churn resultou em um **valor líquido gerado de R$ 363,815.74**.

A análise SHAP aprofunda esse resultado, mostrando que a **otimização da política de fretes** é a principal alavanca para reduzir o churn de forma proativa. Portanto, o projeto entrega uma solução de **duplo valor**: uma **ferramenta tática** para reter clientes em risco e um **mapa estratégico** para atacar a causa raiz do problema.

---

🚧 ## Próximos Passos

Para evoluir este projeto, os próximos passos recomendados são:

1.  **Análise de Sensibilidade:** Avaliar como o valor gerado pelo modelo muda com diferentes taxas de sucesso da campanha de retenção ou custos.
2.  **Deployment:** Implementar o modelo como uma API para que possa ser consumido por outras áreas da empresa e realizar predições em tempo real ou em batch.