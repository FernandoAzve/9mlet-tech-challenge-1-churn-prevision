
"""
Pacote de inferência: **pré-processador sklearn serializado** + **MLP PyTorch** + metadados.

Contrato de pasta (deploy):
- ``preprocessor.joblib`` — ``Pipeline`` sklearn (limpeza + alinhamento + ``StandardScaler``)
- ``mlp_state.pt`` — pesos da rede
- ``metadata.json`` — limiares, dimensões, hash do dataset, etc.

O MLflow continua registrando só a rede PyTorch; o pré-processador fica sempre no bundle local.
"""

from __future__ import annotationsimport jsonfrom pathlib import Pathfrom typing import Anyimport joblibimport numpy as npimport pandas as pdimport torchfrom sklearn.pipeline import Pipelinefrom churn.models.mlp_torch import MLPChurn# Nomes fixos dos arquivos dentro da pasta do bundle (não renomeie à mão em produção).
_PREPROCESSOR_NAME = "preprocessor.joblib"
_WEIGHTS_NAME = "mlp_state.pt"
_METADATA_NAME = "metadata.json"


class ChurnMLPBundle:
    """
    Objeto que encapsula “como passar do DataFrame bruto à probabilidade de churn”.

    Ordem lógica: alinhar colunas → ``preprocessor.transform`` (limpa + escala) →
    rede neural devolve logit → aplicamos sigmoid → número entre 0 e 1.
    """

    def __init__(
        self,
        preprocessor: Pipeline,
        model: MLPChurn,
        feature_columns: list[str],
        threshold_prediction: float,
        threshold_otimo_f1_validacao: float,
        device: torch.device | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        # Lista ordenada dos nomes de colunas usadas no treino (one-hots inclusos).
        self.feature_columns = list(feature_columns)
        # Limiar “padrão” (geralmente 0,5) e o que melhorou F1 na validação.
        self.threshold_prediction = float(threshold_prediction)
        self.threshold_otimo_f1_validacao = float(threshold_otimo_f1_validacao)
        # Expõe nomes de features para a API montar o DataFrame igual ao treino.
        align = preprocessor.named_steps.get("align_features")
        if align is not None and getattr(align, "feature_names_in_", None) is not None:
            self.feature_names_in_ = align.feature_names_in_
        else:
            self.feature_names_in_ = np.array(self.feature_columns, dtype=object)
        self._device = device or torch.device("cpu")
        self.model.to(self._device)
        # eval() desliga dropout/batchnorm de treino; aqui só fazemos predição.
        self.model.eval()

    @classmethod
    def load(cls, bundle_dir: str | Path) -> ChurnMLPBundle:
        """
        Carrega os três arquivos da pasta e monta rede com a mesma arquitetura salva no JSON.
        """
        root = Path(bundle_dir)
        meta_path = root / _METADATA_NAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"Metadata não encontrado: {meta_path}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_columns: list[str] = metadata["feature_columns"]
        input_dim = int(metadata["input_dim"])
        hidden_1 = int(metadata.get("hidden_dim_1", 64))
        hidden_2 = int(metadata.get("hidden_dim_2", 32))

        pre_path = root / _PREPROCESSOR_NAME
        if not pre_path.is_file():
            raise FileNotFoundError(
                f"Pré-processador não encontrado: {pre_path}. "
                "Treine novamente com a versão atual do projeto."
            )
        # Objeto sklearn salvo com joblib (o mesmo que foi .fit no treino).
        preprocessor = joblib.load(pre_path)

        # Recria a arquitetura vazia e injeta os pesos salvos em disco.
        model = MLPChurn(input_dim=input_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2)
        state_path = root / _WEIGHTS_NAME
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

        return cls(
            preprocessor=preprocessor,
            model=model,
            feature_columns=feature_columns,
            threshold_prediction=float(metadata["threshold_prediction"]),
            threshold_otimo_f1_validacao=float(metadata["threshold_otimo_f1_validacao"]),
            device=torch.device("cpu"),
        )

    def save(self, bundle_dir: str | Path, extra_metadata: dict[str, Any] | None = None) -> Path:
        """
        Grava tudo que a API precisa numa pasta: pré-processador, pesos e JSON descritivo.
        ``extra_metadata`` acrescenta campos ao JSON (hash do CSV, contagens, etc.).
        """
        root = Path(bundle_dir)
        root.mkdir(parents=True, exist_ok=True)

        # Pipeline sklearn completo (inclui médias/desvios do StandardScaler).
        joblib.dump(self.preprocessor, root / _PREPROCESSOR_NAME)
        # Só os pesos da rede (state_dict), não o objeto inteiro — mais estável entre versões.
        torch.save(self.model.state_dict(), root / _WEIGHTS_NAME)

        payload: dict[str, Any] = {
            "feature_columns": self.feature_columns,
            "input_dim": len(self.feature_columns),
            "hidden_dim_1": 64,
            "hidden_dim_2": 32,
            "threshold_prediction": self.threshold_prediction,
            "threshold_otimo_f1_validacao": self.threshold_otimo_f1_validacao,
            "preprocessor_file": _PREPROCESSOR_NAME,
            "weights_file": _WEIGHTS_NAME,
        }
        if extra_metadata:
            payload.update(extra_metadata)

        (root / _METADATA_NAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return root

    def predict_proba_positive(self, features: pd.DataFrame) -> np.ndarray:
        """
        Uma probabilidade entre 0 e 1 por linha: “quão forte o modelo aponta para churn”.
        O pré-processador nas etapas iniciais ainda espera DataFrame (não só matriz).
        """
        # Garante mesma ordem de colunas do treino; faltando vira 0 (ex.: one-hot ausente).
        aligned = features.reindex(columns=self.feature_columns, fill_value=0)
        x_s = self.preprocessor.transform(aligned)
        if not isinstance(x_s, np.ndarray):
            x_s = np.asarray(x_s, dtype=np.float32)
        x_s = x_s.astype(np.float32, copy=False)
        x_t = torch.from_numpy(x_s).to(self._device)

        self.model.eval()
        # Sem gradientes: inferência mais rápida e menos memória.
        with torch.no_grad():
            logits = self.model(x_t)
            # Logit → probabilidade com sigmoid (interpretação intuitiva para negócio).
            proba = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        return np.asarray(proba, dtype=np.float64)
