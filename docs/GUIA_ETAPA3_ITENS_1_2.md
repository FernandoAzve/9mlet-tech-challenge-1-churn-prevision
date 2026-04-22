# Guia prático (leigo + técnico) — Etapa 3, Itens 1 e 2

## Objetivo deste guia

Explicar, de forma simples, **como refatorar o projeto para uma estrutura profissional** (Item 1) e **como criar um pipeline reprodutível com sklearn + transformadores custom** (Item 2), sem perder os detalhes técnicos importantes.

---

## Contexto rápido (em linguagem simples)

Hoje seu projeto funciona muito bem em notebooks.  
O próximo passo é transformar isso em um “produto de engenharia”:

- organizado em arquivos com responsabilidades claras;
- fácil de testar;
- fácil de reutilizar na API;
- reproduzível (mesma entrada -> mesmo comportamento esperado).

Pense assim:
- Notebook = laboratório (ótimo para explorar).
- `src/` + pipeline = fábrica (ótimo para operar).

---

## Item 1 — Refatorar código em módulos (`src/`) com estrutura limpa

## Recomendação de tecnologia

- **Python 3.11+**
- **scikit-learn** para pipeline/base de modelo clássico
- **pandas/numpy** para dados
- **joblib** para serializar pipeline/modelo
- **pydantic** (já pensando na API)
- **MLflow** para rastrear treino/experimentos (opcional no primeiro momento da refatoração, mas recomendado)

Se quiser manter o MLP PyTorch depois, sem problema. A recomendação aqui é começar com um pipeline sólido de engenharia usando sklearn para estabilizar o fluxo.

## Estrutura de pastas recomendada

```text
src/
  churn/
    __init__.py
    config.py
    logging_config.py

    data/
      __init__.py
      io.py                # leitura/escrita de datasets
      schemas.py           # contratos de dados (colunas esperadas)

    features/
      __init__.py
      preprocessing.py     # ColumnTransformer e passos padrão
      custom_transformers.py

    models/
      __init__.py
      train.py             # treino
      predict.py           # inferência
      evaluate.py          # métricas
      registry.py          # salvar/carregar artefatos (joblib/mlflow)

    pipelines/
      __init__.py
      churn_pipeline.py    # função que monta pipeline completo

    api/
      __init__.py
      schemas.py           # Pydantic request/response
      service.py           # camada de serviço da inferência
      main.py              # FastAPI app

tests/
  unit/
  schema/
  smoke/
```

## Como eu faria a refatoração (ordem prática)

1. **Congelar contrato de dados atual**  
   Definir claramente: target, colunas de entrada, tipos esperados, colunas proibidas.

2. **Extrair utilitários de leitura e split**  
   Tirar dos notebooks tudo que é repetido (leitura de CSV, split estratificado, seed).

3. **Extrair pré-processamento para módulo de features**  
   Padronizar o pipeline de transformação em uma função única.

4. **Criar módulo de treino com CLI simples**  
   Exemplo: `python -m churn.models.train` gera artefato versionado.

5. **Criar módulo de predição independente de notebook**  
   Exemplo: `python -m churn.models.predict --input arquivo.csv`.

6. **Manter notebook como consumidor do módulo, não como dono da lógica**  
   Notebook passa a importar funções de `src/churn/...`.

Resultado: menos duplicação, menos risco de divergência entre treino e API.

---

## Item 2 — Criar pipeline reprodutível (`sklearn` + transformadores custom)

## O que é “reprodutível” na prática?

É quando você consegue treinar hoje e amanhã com o mesmo código/configuração e obter comportamento consistente (diferenças pequenas podem ocorrer em alguns algoritmos, mas o processo é controlado e auditável).

## Peças técnicas essenciais

1. **`ColumnTransformer`**  
   Permite aplicar transformações diferentes em colunas numéricas e categóricas.

2. **`Pipeline`**  
   Encadeia preprocessing + modelo em um único objeto.  
   Isso evita o erro clássico: transformar no treino de um jeito e na API de outro.

3. **Transformadores custom (`BaseEstimator`, `TransformerMixin`)**  
   Usar quando precisa de regra de negócio específica que não existe pronta no sklearn.

4. **Serialização única (`joblib`)**  
   Salvar o pipeline completo (`preprocess + model`) em um arquivo versionado.

5. **Controle de seed/config**  
   `random_state` fixo e hiperparâmetros centralizados.

## Exemplo de desenho técnico do pipeline

```text
Entrada (DataFrame bruto)
   -> validação mínima de colunas
   -> ColumnTransformer
      - numéricas: imputação + scaling
      - categóricas: imputação + one-hot
      - custom: regras específicas (ex.: limpeza de campos problemáticos)
   -> classificador (ex.: LogisticRegression ou RandomForest)
   -> saída: probabilidade de churn
```

## Exemplo real de transformador custom (conceito)

Você pode criar um transformador para normalizar colunas com problema histórico no seu dataset (ex.: conversão robusta de números em texto).  
Tecnicamente, ele implementa:

- `fit(self, X, y=None)` (normalmente retorna `self`)
- `transform(self, X)` (retorna DataFrame transformado)

Isso encaixa naturalmente dentro do `Pipeline`.

## Como eu faria a implementação do Item 2

1. **Mapear colunas por tipo** (numéricas, categóricas, alvo).  
2. **Definir pré-processamento padrão** em `features/preprocessing.py`.  
3. **Adicionar transformadores custom só onde necessário** (não exagerar).  
4. **Montar função `build_pipeline(config)`** em `pipelines/churn_pipeline.py`.  
5. **Treinar via módulo único** (`models/train.py`) e salvar artefato (`joblib`).  
6. **Testar inferência carregando o artefato** com dados novos.  
7. **Logar parâmetros e métricas no MLflow** para rastreabilidade.

---

## Decisão tecnológica: sklearn agora, PyTorch depois?

Minha recomendação para esta etapa:

- **Etapa 3 (engenharia/API):** priorizar pipeline sklearn para acelerar estabilidade, testes e integração com FastAPI.
- **Evolução posterior:** manter trilha PyTorch como opção avançada (por exemplo, quando você quiser ganho de performance preditiva ou arquitetura neural mais sofisticada).

Isso reduz complexidade no momento em que o foco é engenharia de software e produto.

---

## Riscos comuns e como evitar

- **Risco:** lógica duplicada entre notebook e `src/`.  
  **Mitigação:** notebook só chama funções de `src`.

- **Risco:** data leakage no preprocessing.  
  **Mitigação:** sempre `fit` no treino e apenas `transform` em validação/teste.

- **Risco:** API com pré-processamento diferente do treino.  
  **Mitigação:** API carrega exatamente o mesmo artefato do pipeline treinado.

- **Risco:** “funciona na minha máquina”.  
  **Mitigação:** config centralizada + comandos padronizados + testes.

---

## Definição de pronto (Itens 1 e 2)

Você pode considerar os itens concluídos quando:

1. Existe estrutura `src/` organizada e sem dependência de notebook para inferência.
2. Pipeline sklearn completo (preprocess + modelo) está em código modular.
3. Artefato versionado do pipeline é salvo e recarregado com sucesso.
4. A mesma função de inferência pode ser usada por script e API.
5. O fluxo está documentado de forma que outra pessoa do time consiga executar.

---

## Resumo em uma frase

Eu faria a Etapa 3 começando por uma **espinha dorsal de engenharia limpa em `src/`** e um **pipeline sklearn único e serializável**, para garantir consistência entre treino e produção antes de aumentar a complexidade do modelo.
