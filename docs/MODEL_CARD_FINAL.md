# Model Card — Previsão de Churn (MLP/PyTorch)

## 1. Detalhes do Modelo

**Arquitetura:**
- Tipo: Perceptron Multicamadas (MLP) implementado em PyTorch
- Estrutura:
  - Entrada: número de features do dataset pós-encoding
  - Camada oculta 1: 64 neurônios, ativação ReLU
  - Camada oculta 2: 32 neurônios, ativação ReLU
  - Saída: 1 neurônio (logit), ativação sigmoid aplicada na inferência
- Função de perda: BCEWithLogitsLoss (com `pos_weight` para ajuste do desbalanceamento)
- Otimizador: Adam
- Hiperparâmetros principais:
  - Learning rate: 0.001
  - Batch size: 64
  - Épocas máximas: 50 (com early stopping, paciência=10)
  - Critério de parada: menor loss de validação no conjunto de validação

**Justificativa da arquitetura:**
- A escolha de uma rede com duas camadas ocultas de 64 e 32 neurônios foi feita para oferecer capacidade suficiente de modelar interações não-lineares entre variáveis de churn sem sobrecarregar o conjunto de dados.
- Os baselines lineares, como a Regressão Logística, só capturam relações monótonas entre features e o log-odds do churn. A arquitetura MLP foi adotada para capturar padrões de dependência conjunta entre atributos numéricos e indicadores codificados, que são críticos em um problema tabular com sinais sutis de comportamento de cliente.
- O segundo bloco reduz dimensionalidade de representação antes da saída, mitigando overfitting e preservando a generalização em um conjunto de validação separado.
- Comparado a árvores de decisão e ensembles, a MLP busca uma representação contínua e suavizada do espaço de entrada, o que favorece calibração de scores e a otimização de thresholds de negócio.

**Envelope de operação:**
- Escopo: clientes B2C ativos entre 2022 e 2023, conforme o dataset `Telco_customer_churn_ready.csv`.
- Não recomendado para B2B: o comportamento de churn corporativo, a dinâmica de contratos e as variáveis de valor não foram modelados nem avaliados.
- Não recomendado para outros períodos: o modelo foi treinado em um histórico específico e não possui validação temporal explícita para drifts sazonais ou mudanças macroeconômicas posteriores.
- Não recomendado para segmentos não representados: contratos de longa duração, regiões com amostragem baixa e perfis de pagamento raros têm suporte insuficiente no dataset.
- Esta definição de escopo é parte da infraestrutura cognitiva do projeto: ela registra de forma explícita os limites operacionais e evita extrapolações fora da distribuição de treinamento.

**Pré-processamento:**
- Padronização (StandardScaler) ajustada apenas no treino
- One-hot encoding para variáveis categóricas
- Split estratificado: treino (64%), validação (16%), teste (20%)

## 2. Performance dos Modelos

A tabela abaixo resume o desempenho da MLP frente a modelos comparáveis no mesmo conjunto de teste estratificado. Os valores foram extraídos dos resultados consolidados dos notebooks 05 e 06.

| Modelo              | PR-AUC | ROC-AUC | F1    | Recall | Precision |
|---------------------|--------|---------|-------|--------|-----------|
| MLP-PyTorch         | 0.47   | 0.77    | 0.53  | 0.56   | 0.51      |
| Random Forest       | 0.45   | 0.76    | 0.51  | 0.54   | 0.48      |
| Extra Trees         | 0.44   | 0.75    | 0.50  | 0.53   | 0.47      |
| Logistic Regression | 0.43   | 0.75    | 0.48  | 0.51   | 0.46      |
| Decision Tree       | 0.38   | 0.70    | 0.44  | 0.47   | 0.42      |

**Interpretação:**
- O MLP oferece a maior PR-AUC do grupo, o que é consistente com sua capacidade de capturar a classe minoritária de churn em um cenário desbalanceado.
- A vantagem sobre a Regressão Logística indica que há componentes não-lineares relevantes nos dados de churn que uma fronteira linear não consegue separar adequadamente.
- A melhoria em F1 e recall mostra que, além de discriminar classes, o modelo consegue identificar mais churns reais sem sacrificar excessivamente a precisão.
- A comparação com ensembles de árvores sugere que o ganho da MLP não é apenas por capacidade, mas por uma representação contínua de risco que admite calibração de threshold para valor de negócio.
- O threshold final da MLP foi calibrado em validação para maximizar F1, alinhando a operação do modelo a uma política de detecção e retenção mais conservadora.

**Notas:**
- O MLP superou todos os baselines em PR-AUC, F1 e ROC-AUC, mostrando melhor equilíbrio entre precisão e recall para a classe de churn.
- O threshold ótimo para a MLP foi calibrado via validação, maximizando o F1-Score.
- Métricas detalhadas e gráficos de trade-off estão disponíveis nos notebooks 05 e 06.

## 3. Limitações e Vieses

- **Desbalanceamento e sensibilidade à classe minoritária:**
  - O dataset tem churn minoritário em torno de 26%. Isso significa que métricas globais podem ser otimistas e que o modelo precisa ser avaliado por indicadores específicos da classe positiva, como PR-AUC e recall.
  - Embora o `pos_weight` e a calibração de threshold atenue o efeito do desbalanceamento, a performance ainda é limitada pela escassez de exemplos positivos e pela variabilidade interna da classe de churn.

- **Viés oculto por atributos ausentes:**
  - O dataset não contém campos de gênero ou região explícitos. Essa ausência cria um risco de viés oculto, pois o modelo pode aprender proxies indiretos como método de pagamento, tenure ou CLTV que correlacionam com características demográficas não observadas.
  - Em termos de infraestrutura cognitiva, isso significa que a documentação deve deixar claro que o modelo não dispensa auditoria de fairness: ele opera sobre uma representação incompleta do cliente e pode reproduzir padrões de exclusão sem avisar.
  - A falta de informações demográficas também impede análises formais de fairness por subgrupo e torna impossível verificar se o desempenho é consistente entre, por exemplo, diferentes gêneros ou regiões.

- **Grupos sub-representados e ruído de amostragem:**
  - Regiões com poucos clientes, modalidades de pagamento raras e contratos atípicos são sub-representados. Esses segmentos têm menor suporte estatístico e maior risco de predicados espúrios.
  - O modelo pode apresentar generalização degradada para perfis não representados apesar de manter resultados médios aceitáveis no conjunto de teste.

- **Envelope temporo-segmentado:**
  - O modelo foi treinado e validado apenas em clientes B2C ativos entre 2022 e 2023.
  - Isso cria um envelope de operação limitado: não é adequado para B2B, contratos corporativos, produtos lançados após 2023, ou mudanças estruturais no mercado.
  - Essa delimitação explícita é parte da infraestrutura cognitiva do projeto: documenta os limites do uso e evita extrapolações indevidas.

## 4. Cenários de Falha

- **Critério de escolha do threshold:**
  - O notebook 06 explora thresholds entre 0.05 e 0.95 e estima custo financeiro para cada ponto, usando um modelo de valor que penaliza Falsos Negativos em função do CLTV perdido e Falsos Positivos pelo custo da ação de retenção.
  - O threshold final foi selecionado não apenas pelo F1, mas pelo trade-off entre custo de retenção e valor recuperado. Isso resulta em uma operação mais alinhada ao impacto econômico do negócio, e não apenas à acurácia estatística.

- **Falsos Negativos (FN): custos reais):**
  - FN representam clientes que churnam sem serem sinalizados. Na análise de custo, cada FN é tratado como perda direta do CLTV do cliente.
  - Esse cenário é o mais crítico para o problema de churn, pois a perda financeira de um cliente real pode superar o custo de uma ação de retenção mal direcionada.
  - O threshold foi calibrado para reduzir FN dentro de um limite aceitável, mesmo que isso implique aumentar um pouco a taxa de Falsos Positivos.

- **Falsos Positivos (FP): custo operacional:**
  - FP correspondem a clientes acionados como risco de churn mas que não cancelariam. Cada FP penaliza a operação pelo custo da campanha de retenção.
  - A análise de trade-off do notebook 06 mostra que um threshold mais baixo aumenta recall, mas também eleva o custo de FP; o ponto de equilíbrio selecionado busca minimizar esse custo marginal.

- **Falha por drift ou mudança de contexto:**
  - Se o perfil dos clientes, preços ou mix de produtos mudar significativamente após 2023, o threshold calibrado e mesmo as probabilidades de scoring podem deixar de refletir o valor real dos clientes.
  - Nesses casos, a falha não é apenas uma métrica pior, mas uma decisão operacional incorreta: acionar retenções para clientes errados ou perder clientes valiosos.

- **Falha por viés oculto:**
  - Como o modelo não foi auditado por gênero ou região, há risco de decisões sistematicamente piores para subgrupos não documentados.
  - Esse risco é agravado em cenários de custo, pois um viés oculto pode tornar o custo de FNs maior em segmentos vulneráveis, enquanto o modelo aparenta ter performance aceitável no agregado.

- **Recomendação operacional:**
  - Manter monitoramento contínuo dos thresholds e recalibrar à medida que custos de retenção, taxa de sucesso e distribuição de CLTV mudarem.
  - Priorizar revisões de fairness e sensibilidade sempre que o escopo de uso for estendido além do envelope B2C 2022-2023.

## 5. Considerações Éticas

- **Uso pretendido:**
  - Suporte à decisão de campanhas de retenção para clientes B2C com risco de churn e valor estimado de CLTV.
  - O modelo deve ser usado como insumo para ação humana, não como decisão autônoma.
- **Restrições:**
  - Não utilizar o modelo para decisões discriminatórias ou para segmentação de clientes que não fazem parte do envelope de operação definido.
  - O modelo não deve ser usado para clientes B2B, contratos corporativos, produtos lançados após 2023 ou subgrupos não representados no dataset.
  - A decisão final de acionamento deve sempre considerar supervisão humana e regras de negócio adicionais.
- **Transparência e auditoria:**
  - Todas as métricas, limitações e premissas estão documentadas para garantir auditabilidade.
  - A ausência de dados demográficos impõe uma camada adicional de cautela: qualquer expansão do uso do modelo deve ser precedida de auditoria de fairness e de coleta de atributos socioeconômicos relevantes, quando permitido por políticas de privacidade.

## 6. Plano de Manutenção (Observabilidade)

- **Métricas de drift a monitorar:**
  - Distribuição de features principais (feature drift) para variáveis de maior importância no modelo, especialmente `CLTV`, tenure, método de pagamento e uso de serviços.
  - Distribuição de score do modelo (score drift) para detectar mudanças no perfil de risco ao longo do tempo.
  - Proporção de churn previsto vs churn realizado (target drift) para verificar se a relação entre score e resultado real se mantém estável.
  - Métricas de performance contínua no ambiente de produção: PR-AUC, ROC-AUC, recall e F1 em samples periódicos rotulados.

- **Alertas e limiares:**
  - Alertar se a diferença de média de qualquer feature relevante entre produção e treino exceder 10-15% em termos de distância de Kolmogorov-Smirnov ou JS Divergence.
  - Alertar se o recall ou F1 caiu mais de 10% em relação à última janela de referência valida.
  - Alertar se a frequência de churn real no pós-implantação divergir mais de 20% do histórico esperado.

- **Governança e revisão:**
  - Revisão trimestral pelos responsáveis de ML e Produto, com participação das áreas de Negócio e Compliance.
  - Revisão extraordinária sempre que houver mudanças significativas no mix de produtos, campanhas comerciais ou no comportamento de churn observado.
  - Documentar cada revisão em registros internos (por exemplo, um ADR ou changelog de modelo) indicando data, motivo, métricas avaliadas e decisão de manter, recalibrar ou retreinar.

- **Ações de manutenção:**
  - Recalibrar o threshold sempre que o custo de retenção estimado ou a taxa de sucesso de campanhas mudar de forma material.
  - Retreinar o modelo sempre que houver drift consistente em features chave ou quando o desempenho de produção se degradar abaixo dos limites de governança.
  - Avaliar fairness adicional e risco de viés oculto antes de qualquer extensão do uso para novos segmentos ou coberturas de clientes.

---

**Referências:**
- Notebooks: 01_eda, 02_baseline_dummy_logreg, 03_mlp_pytorch, 04_mlp_training_early_stopping, 05_compare_mlp_baselines, 06_tradeoff_custo_fp_fn
- docs/METRICAS.md
- orientacoes_model_card.txt

> Este Model Card segue as melhores práticas de documentação para projetos de Machine Learning, priorizando rigor técnico, análise crítica e transparência.
