# Definição e Justificativa das Métricas de Avaliação

Neste projeto de predição de churn para a operadora de telecomunicações, um dos principais desafios é lidar com a natureza desbalanceada dos dados. Como a grande maioria dos clientes permanece na base, o uso de métricas simples (como a acurácia) pode gerar uma falsa impressão de bom desempenho. Afinal, um modelo que simplesmente dissesse que 'ninguém vai cancelar' acertaria muito, mas não resolveria o problema do negócio.

Por conta disso, estruturamos a avaliação do nosso modelo (tanto as baselines quanto a rede neural em PyTorch) em duas frentes: **avaliação técnica** e **impacto de negócio**.

---

## 1. Métricas Técnicas

O foco técnico é garantir que o algoritmo esteja realmente aprendendo os padrões de quem cancela, sem ser enganado pela classe majoritária. Para isso, acompanhamos via MLflow os seguintes indicadores:

- **PR-AUC (Area Under the Precision-Recall Curve):** 
Esta é eleita a nossa **métrica técnica principal**. Quando lidamos com conjuntos desbalanceados, a curva PR foca especificamente na classe minoritária (o churn). Ela nos mostra o quão bem o modelo consegue manter a qualidade dos alertas à medida que tenta capturar mais cancelamentos, penalizando falsos alarmes de forma mais realista que a curva ROC.

- **F1-Score:** 
É a média harmônica entre a Precisão (dos que classificamos como churn, quantos acertamos) e o Recall (de todos que realmente deram churn, quantos encontramos). Nós a utilizamos como nossa 'balança'. O F1-Score é ótimo para nos guiar no momento de escolher o limiar de decisão (ponto de corte) da rede neural, buscando um meio-termo para não perder clientes de vista e nem onerar o marketing com campanhas desnecessárias.

- **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** 
Mantemos o ROC-AUC como uma métrica de apoio. Ela nos dá uma leitura geral da capacidade discriminativa do modelo, ou seja, se pegarmos um cliente qualquer que cancelou e um que não cancelou, o quão capaz a rede é de dar a pontuação de risco maior para a pessoa correta.

---

## 2. Métrica de Negócio 

Para apresentar o projeto aos tomadores de decisão, as métricas estatísticas não bastam. A diretoria precisa saber o cruzamento desse modelo com a operação real.

- **Custo do Churn Evitado (Retorno Financeiro):**
Em vez de avaliar apenas acertos e erros absolutos, avaliamos o impacto financeiro da matriz de confusão. A lógica consiste em atribuir um custo real para cada decisão:

  **1. Verdadeiro Positivo:** O modelo alerta o risco, a operadora aciona uma oferta (que tem um custo), e o cliente fica. Retemos a receita desse cliente (CLTV) subtraindo apenas o custo da ação.
  **2. Falso Positivo:** O modelo erra e prevemos churn em um cliente fiel. Acabamos gastando dinheiro ativando campanhas de retenção para uma pessoa que não precisava.
  **3. Falso Negativo:** O pior cenário. O modelo não detecta, não agimos, e o cliente vai embora, resultando na perda total daquela receita.

O objetivo da métrica de negócio é encontrar o cenário onde a soma do capital salvo (Verdadeiros Positivos) seja expressivamente superior às perdas e custos de campanhas mal direcionadas (Falsos Negativos e Positivos).

---

## Qual métrica priorizar? A abordagem conjunta

Em um cenário real de MLOps, a abordagem mais adequada é **utilizar as três**, mas em momentos distintos da esteira de desenvolvimento:

1. Durante o treinamento e otimização dos pesos, os engenheiros observam o **PR-AUC** para selecionar a melhor arquitetura e hiperparâmetros, pois ele reflete fielmente a detecção da classe rara.
2. Com o modelo base escolhido, utilizamos o simulador do **Custo de Churn Evitado** atrelado ao **F1-Score** para calibrar a nota de corte final que vai para produção. O modelo ideal para o negócio entregue na nossa API será aquele com a calibragem que estipule o maior lucro possível para a telecom.
