"""Treino do modelo único a partir de (sinal padrão v3, saída da cadeia).

Usa as peças do `nam.train` (datasets, LightningModule, splits do v3), mas com
uma WaveNet clássica em vez do PackedWaveNet do treinador simplificado 0.13.
"""

from . import _compat  # noqa: F401

import hashlib as _hashlib
import json as _json
import logging as _logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as _pl
import torch
from torch.utils.data import DataLoader as _DataLoader

from nam.data import Split as _Split
from nam.data import apply_joint_dataset_hooks as _apply_joint_hooks
from nam.data import get_joint_dataset_hooks as _get_joint_hooks
from nam.data import init_dataset as _init_dataset
from nam.train import core as _core
from nam.train import full as _full
from nam.train.lightning_module import LightningModule as _LightningModule
from nam.train.lightning_module import PackedBestCheckpoint as _PackedBestCheckpoint
from nam.train.lightning_module import PackedLightningModule as _PackedLightningModule
from nam.util import filter_warnings as _filter_warnings

from .archs import is_packed, lightning_model_config
from .nam_format import UnsupportedForLegacyExport, to_legacy_a1

logger = _logging.getLogger(__name__)

V3_MD5 = "36cd1af62985c2fac3e654333e36431e"
V3_URL = (
    "https://drive.google.com/file/d/1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9/view"
)

# Faixas de ESR da spec (§9 passo 6).
ESR_GREAT = 0.01
ESR_FAIL = 0.05
# Alvo da parada automática. Medido (TS-808 -> FRUSCIANTE, A2): começando do amp
# o treino chega aqui em ~16 épocas; do zero, nem 0.01 em 20 épocas.
ESR_AUTO_STOP = 0.005


def md5(path: Union[str, Path]) -> str:
    h = _hashlib.md5()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_v3_input(path: Union[str, Path]) -> None:
    got = md5(path)
    if got != V3_MD5:
        raise ValueError(
            f"{path} não é o v3_0_0.wav oficial (md5 {got}). Baixe em {V3_URL}"
        )


def esr_verdict(esr: float) -> str:
    if esr < ESR_GREAT:
        return "ótimo"
    if esr <= ESR_FAIL:
        return "ok"
    return "ruim"


def _drop_nulls(v):
    if isinstance(v, dict):
        return {k: _drop_nulls(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [_drop_nulls(x) for x in v if x is not None]
    return v


@dataclass
class TrainResult:
    nam_path: Path
    legacy_nam_path: Optional[Path]
    esr: float
    epochs_run: int
    seconds: float
    esr_by_size: Optional[list] = None
    warm_start: Optional[str] = None
    stop_reason: Optional[str] = None


def _quality_esr(trainer) -> Optional[float]:
    """ESR de validação do modelo que vai para o pedal (o maior, no A2)."""
    m = trainer.callback_metrics
    packed = sorted(k for k in m if k.startswith("ESR_packed_"))
    key = packed[-1] if packed else ("ESR" if "ESR" in m else "val_loss")
    v = m.get(key)
    return None if v is None else float(v)


class ProgressFile(_pl.Callback):
    """Grava o progresso em JSON a cada validação: época, ESR atual/melhor e histórico."""

    def __init__(self, path: Union[str, Path], warm_start: Optional[str] = None):
        self.path = Path(path)
        self.best: Optional[float] = None
        self.history: list = []
        self.warm_start = warm_start

    def _write(self, trainer, val: Optional[float]) -> None:
        if val is not None:
            self.best = val if self.best is None else min(self.best, val)
            self.history.append(round(val, 6))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(_json.dumps({
            "epoch": int(trainer.current_epoch),
            "max_epochs": int(trainer.max_epochs or 0),
            "val_loss": val,
            "best_val_loss": self.best,
            "esr_history": self.history,
            "warm_start": self.warm_start,
        }))
        tmp.replace(self.path)

    def on_train_start(self, trainer, pl_module):
        self._write(trainer, None)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._write(trainer, _quality_esr(trainer))


class AutoStop(_pl.Callback):
    """Para o treino quando ele já chegou lá, em vez de sempre ir até o fim.

    - `target_esr`: o modelo do pedal chegou a esse ESR (bem abaixo do "ótimo" 0.01);
    - platô: o melhor ESR não melhorou `rel_delta` (relativo) em `patience` épocas.
    Nunca para antes de `min_epochs`.
    """

    def __init__(self, target_esr: Optional[float] = ESR_AUTO_STOP, patience: int = 10,
                 rel_delta: float = 0.03, min_epochs: int = 8):
        self.target_esr = target_esr
        self.patience = patience
        self.rel_delta = rel_delta
        self.min_epochs = min_epochs
        self.best: Optional[float] = None
        self.since_best = 0
        self.reason: Optional[str] = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        esr = _quality_esr(trainer)
        if esr is None:
            return
        if self.best is None or esr < self.best * (1 - self.rel_delta):
            self.best = esr
            self.since_best = 0
        else:
            self.best = min(self.best, esr)
            self.since_best += 1
        epochs_done = trainer.current_epoch + 1
        if epochs_done < self.min_epochs:
            return
        if self.target_esr is not None and esr <= self.target_esr:
            self.reason = f"chegou ao ESR {esr:.4f} (alvo {self.target_esr})"
        elif self.since_best >= self.patience:
            self.reason = f"parou de melhorar há {self.patience} épocas (ESR {self.best:.4f})"
        if self.reason:
            logger.info("Parando o treino: %s", self.reason)
            trainer.should_stop = True


# ---------------------------------------------------------------- warm start


def _container_submodels(d: Dict[str, Any]) -> list:
    """Submodelos de um container, do menor para o maior (como no PackedWaveNet)."""
    subs = sorted(d["config"]["submodels"], key=lambda s: s.get("max_value", 0.0))
    out = []
    for s in subs:
        m = dict(s["model"])
        m.setdefault("sample_rate", d.get("sample_rate"))
        out.append(m)
    return out


def _match_head_scale(src, dst_scale: float) -> bool:
    """Ajusta `src` para ter o `head_scale` do modelo de destino, sem mudar o som.

    A saída é `head_scale * head_rechannel(...)`, linear nos pesos da última
    camada: multiplicar peso e bias pela razão dá exatamente a mesma saída.
    Capturas do Tone3000 trazem o head_scale aprendido (ex.: 0.0116) e o
    treinador usa 0.01 fixo.
    """
    net = src._net
    src_scale = float(getattr(net, "_head_scale", dst_scale))
    if abs(src_scale - dst_scale) < 1e-12:
        return True
    arrays = getattr(net, "_layer_arrays", None)
    if arrays is None or len(arrays) != 1 or not hasattr(arrays[0], "_head_rechannel"):
        return False
    r = src_scale / dst_scale
    with torch.no_grad():
        for p in arrays[0]._head_rechannel.parameters():
            p.mul_(r)
    net._head_scale = dst_scale
    return True


def warm_start(model, packed: bool, nam_path: Union[str, Path]) -> bool:
    """Começa o treino com os pesos de uma captura da própria cadeia.

    A saída da cadeia costuma ser parecida com a de um dos seus estágios (o amp),
    então partir dele em vez de pesos aleatórios chega no mesmo ESR em bem menos
    épocas. Só vale se a captura tem exatamente a mesma estrutura do modelo que
    vamos treinar; se não, devolve False e o treino começa do zero.
    """
    from nam.models import init_from_nam

    from .nam_format import upgrade_model_dict

    d = _json.loads(Path(nam_path).read_text())
    try:
        if packed:
            if d.get("architecture") != "SlimmableContainer":
                return False
            subs = _container_submodels(d)
            if len(subs) != model.net.num_submodels:
                return False
            srcs = []
            for i, sub in enumerate(subs):
                src = init_from_nam(upgrade_model_dict(sub))
                dst = model.net.extract_submodel(i)
                if [p.shape for p in src.parameters()] != [p.shape for p in dst.parameters()]:
                    return False
                if not _match_head_scale(src, float(dst._net._head_scale)):
                    return False
                srcs.append(src)
            for i, src in enumerate(srcs):
                model.net.import_submodel(i, src)
        else:
            if d.get("architecture") == "SlimmableContainer":
                d = _container_submodels(d)[-1]
            src = init_from_nam(upgrade_model_dict(d))
            ss, ds = src.state_dict(), model.net.state_dict()
            if ss.keys() != ds.keys() or any(ss[k].shape != ds[k].shape for k in ss):
                return False
            if not _match_head_scale(src, float(getattr(model.net._net, "_head_scale", 0.02))):
                return False
            model.net.load_state_dict(src.state_dict())
    except Exception as e:  # captura estranha: treina do zero
        logger.warning("Não deu para começar de %s: %s", nam_path, e)
        return False
    logger.info("Treino começa a partir de %s", Path(nam_path).name)
    return True


def _device_config() -> Dict[str, Any]:
    if torch.cuda.is_available():
        return {"accelerator": "gpu", "devices": 1}
    if torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    logger.warning("Sem GPU: o treino vai ser MUITO lento (use o Colab)")
    return {"accelerator": "cpu"}


def train_chain_model(
    input_wav: Union[str, Path],
    output_wav: Union[str, Path],
    outdir: Union[str, Path],
    arch: str = "a2",
    epochs: int = 100,
    batch_size: int = 16,
    ny: int = _core._NY_DEFAULT,
    seed: int = 0,
    basename: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
    legacy_export: bool = True,
    skip_input_check: bool = False,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    progress_path: Optional[Union[str, Path]] = None,
    init_from: Optional[Union[str, Path, list]] = None,
    auto_stop: bool = True,
    target_esr: Optional[float] = ESR_AUTO_STOP,
) -> TrainResult:
    """Treina e exporta `<basename>.nam`.

    - arch "a2": SlimmableContainer (0.7.0), como o treinador oficial e o Tone3000.
    - archs clássicas (A1): WaveNet 0.7.0 + `<basename>.a1.nam` no formato 0.5.4.
    - `init_from`: captura(s) da cadeia para começar dela (ver `warm_start`);
      tenta na ordem e usa a primeira com a mesma estrutura.
    - `auto_stop`: para antes de `epochs` quando o ESR chega ao alvo ou estabiliza.
    """
    from time import time

    t0 = time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not skip_input_check:
        check_v3_input(input_wav)
    torch.manual_seed(seed)

    # Splits oficiais do v3. Latência 0: a "reamp" é virtual e já alinhada.
    data_config = _core._get_data_config(
        input_version=_core._Version(3, 0, 0),
        input_path=str(input_wav),
        output_path=str(output_wav),
        ny=ny,
        latency=0,
    )
    model_config = lightning_model_config(arch)
    learning_config = {
        "train_dataloader": {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": torch.cuda.is_available(),
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {},
        "trainer": {"max_epochs": epochs, **_device_config(), **(trainer_kwargs or {})},
    }
    for name, cfg in (("data", data_config), ("model", model_config), ("learning", learning_config)):
        (outdir / f"config_{name}.json").write_text(_json.dumps(cfg, indent=2))

    packed = is_packed(arch)
    lightning_cls = _PackedLightningModule if packed else _LightningModule
    model = lightning_cls.init_from_config(model_config)
    data_config["common"]["nx"] = model.net.receptive_field
    ds_train = _init_dataset(data_config, _Split.TRAIN)
    ds_val = _init_dataset(data_config, _Split.VALIDATION)
    _apply_joint_hooks(
        dataset_train=ds_train,
        dataset_validation=ds_val,
        hooks=_get_joint_hooks(data_config.get("joint", [])),
    )
    model.net.sample_rate = ds_train.sample_rate
    _full._handshake_datasets(model, ds_train, ds_val)
    started_from = None
    candidates = [] if init_from is None else (
        [init_from] if isinstance(init_from, (str, Path)) else list(init_from))
    for cand in candidates:  # a primeira que encaixa na estrutura do modelo
        if warm_start(model, packed, cand):
            started_from = Path(cand).name
            break

    callbacks = _full._create_callbacks(learning_config, packed=packed)
    extra = []
    if progress_path:
        extra.append(ProgressFile(progress_path, warm_start=started_from))
    stopper = AutoStop(target_esr=target_esr) if auto_stop else None
    if stopper:
        extra.append(stopper)
    trainer = _pl.Trainer(
        callbacks=callbacks + extra,
        default_root_dir=str(outdir),
        enable_progress_bar=False,
        # Sem TensorBoard: não usamos os logs e o app desktop não empacota o pacote.
        logger=False,
        **learning_config["trainer"],
    )
    with _filter_warnings("ignore", category=UserWarning):
        trainer.fit(
            model,
            _DataLoader(ds_train, **learning_config["train_dataloader"]),
            _DataLoader(ds_val, **learning_config["val_dataloader"]),
        )

    best = trainer.checkpoint_callback.best_model_path
    if best:
        model = lightning_cls.load_from_checkpoint(
            best, **lightning_cls.parse_config(model_config)
        )
    model.cpu()
    model.eval()
    model.net.sample_rate = ds_train.sample_rate
    if packed:
        # Como o treinador oficial: cada tamanho fica com o seu melhor checkpoint.
        pb = next((c for c in callbacks if isinstance(c, _PackedBestCheckpoint)), None)
        if pb is not None:
            _core._apply_packed_best_checkpoints(model, model_config, pb.checkpoint_paths)

    with torch.no_grad():
        pred = model(ds_val.x)
    esr_by_size = None
    if packed:
        out = pred.reshape(-1, pred.shape[-1])  # (tamanhos, amostras)
        esr_by_size = [_core._esr(o, ds_val.y) for o in out]
        esr = esr_by_size[-1]  # o tamanho de maior qualidade
    else:
        esr = _core._esr(pred.flatten(), ds_val.y)

    md = dict(metadata or {})
    md["chain_capture"] = {
        **md.get("chain_capture", {}),
        "arch": arch,
        "esr": esr,
        "epochs": trainer.current_epoch,
    }
    if started_from:
        md["chain_capture"]["warm_start"] = started_from
    if esr_by_size is not None:
        md["chain_capture"]["esr_by_size"] = esr_by_size
    # Leitores de pedaleira são mais simples que o plugin: sem valores null.
    model.net.export(outdir, basename=basename, other_metadata=_drop_nulls(md))
    nam_path = outdir / f"{basename}.nam"

    legacy_path: Optional[Path] = None
    if legacy_export and not packed:
        try:
            legacy = to_legacy_a1(_json.loads(nam_path.read_text()))
            legacy_path = outdir / f"{basename}.a1.nam"
            legacy_path.write_text(_json.dumps(legacy))
        except UnsupportedForLegacyExport as e:  # pragma: no cover
            logger.warning("Sem export legado: %s", e)

    for ds in (ds_train, ds_val):
        if hasattr(ds, "teardown"):
            ds.teardown()
    return TrainResult(
        nam_path=nam_path,
        legacy_nam_path=legacy_path,
        esr=esr,
        epochs_run=trainer.current_epoch,
        seconds=time() - t0,
        esr_by_size=esr_by_size,
        warm_start=started_from,
        stop_reason=stopper.reason if stopper else None,
    )
