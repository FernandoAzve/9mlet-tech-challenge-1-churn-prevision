# Relatório pré-mudanças dos notebooks

## Escopo avaliado
- `notebooks/01_eda.ipynb`
- `notebooks/02_baseline_dummy_logreg.ipynb`
- `notebooks/03_mlp_pytorch.ipynb`
- `notebooks/04_mlp_training_early_stopping.ipynb`
- `notebooks/05_compare_mlp_baselines.ipynb`
- `notebooks/06_tradeoff_custo_fp_fn.ipynb`

## Critérios de nota (0 a 10)
- Portabilidade (35%): caminhos relativos e independência de máquina.
- Robustez de execução (35%): warnings/erros na execução.
- Legibilidade (20%): comentários explicativos em blocos críticos.
- Organização/reprodutibilidade (10%): setup claro e execução consistente.

## Diagnóstico geral antes das mudanças
- Havia ocorrência de caminhos absolutos em *outputs* salvos (ex.: `D:\...` e `C:\Users\...`), mesmo com código majoritariamente relativo.
- Havia warnings de execução em notebooks com MLflow e de convergência em baseline logística.
- Alguns trechos críticos tinham pouca explicação local de fluxo (principalmente em células mais longas de modelagem/avaliação).
- A execução fim a fim era possível, mas com ruído de warning e baixa portabilidade visual dos resultados.

## Achados e notas por notebook (antes)

### `notebooks/01_eda.ipynb`
- Estado: sem warning funcional relevante, porém com output contendo caminho absoluto ao salvar dataset codificado.
- Nota pré: **7.8/10**

### `notebooks/02_baseline_dummy_logreg.ipynb`
- Estado: print de caminho absoluto no dataset e warning de convergência/ruído de paralelização em ambiente Windows durante execução.
- Nota pré: **7.2/10**

### `notebooks/03_mlp_pytorch.ipynb`
- Estado: estrutura limpa, mas com output contendo caminho absoluto no carregamento do dataset.
- Nota pré: **7.6/10**

### `notebooks/04_mlp_training_early_stopping.ipynb`
- Estado: warning em tracking/log de MLflow e output com caminho absoluto de dataset.
- Nota pré: **6.8/10**

### `notebooks/05_compare_mlp_baselines.ipynb`
- Estado: maior concentração de warnings (MLflow) e output de dataset com caminho absoluto.
- Nota pré: **6.2/10**

### `notebooks/06_tradeoff_custo_fp_fn.ipynb`
- Estado: warnings de MLflow e exposição de caminho absoluto em output de diretório temporário de modelo.
- Nota pré: **6.5/10**

## Resumo pré-mudanças
- Média das notas: **7.0/10**.
- Principal risco: portabilidade e rastreabilidade visual por conta de paths absolutos em outputs.
- Principal oportunidade: reduzir ruído de warning e melhorar explicação em células críticas.
