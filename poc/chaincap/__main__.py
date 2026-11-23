"""CLI do PoC.

    python -m chaincap render chain.json --input v3_0_0.wav --out out/
    python -m chaincap run    chain.json --input v3_0_0.wav --out out/ [--arch feather]

`chain.json`:
    {
      "name": "Drive + Amp + Cab",
      "arch": "standard",
      "stages": [
        {"type": "nam", "file": "drive.nam", "input_gain_db": 0, "output_gain_db": 0},
        {"type": "nam", "file": "amp.nam"},
        {"type": "ir",  "file": "cab.wav", "ir_max_ms": 20}
      ]
    }
Divisão (dois amps em paralelo, somados no mixer):
    {"type": "split", "branches": [
        {"level_db": 0, "invert": false, "stages": [{"type": "nam", "file": "ampA.nam"}]},
        {"level_db": -3, "invert": true, "stages": [{"type": "nam", "file": "ampB.nam"}]}
    ]}
Caminhos relativos em `stages[].file` são resolvidos a partir do chain.json.
"""

from . import _compat  # noqa: F401

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from .archs import ARCHITECTURES
from .chain import (
    SAMPLE_RATE,
    ChainResult,
    parse_node,
    read_wav_mono,
    render_chain,
    warm_start_candidates,
    write_wav_24,
)

# Versão do "motor" de render: mude quando o som de uma mesma cadeia mudar,
# para não reaproveitar renders antigos do cache.
RENDER_VERSION = 1
CACHE_KEEP = 4


def _load_chain(path: Path):
    spec = json.loads(path.read_text())
    stages = [parse_node(s, base_dir=path.parent) for s in spec["stages"]]
    return spec, stages


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_key(spec: dict, stages, input_path: Path) -> str:
    """Identifica o render: cadeia (ganhos, ordem...), conteúdo dos arquivos e sinal."""
    files = set()
    for n in stages:
        for st in (s for b in n.branches for s in b.stages) if hasattr(n, "branches") else [n]:
            files.add(Path(st.file))
    h = hashlib.sha256()
    h.update(json.dumps({"v": RENDER_VERSION, "stages": spec["stages"]}, sort_keys=True).encode())
    for f in sorted(files, key=str):
        h.update(f.name.encode() + _file_digest(f).encode())
    h.update(_file_digest(input_path).encode())
    return h.hexdigest()[:24]


def _cache_prune(cache: Path) -> None:
    entries = sorted(cache.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in entries[CACHE_KEEP:]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)


def _render(args) -> tuple:
    spec, stages = _load_chain(Path(args.chain))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "target.wav"
    t0 = time.time()

    cache = Path(args.cache_dir) if getattr(args, "cache_dir", None) else None
    key = _render_key(spec, stages, Path(args.input)) if cache else None
    if cache and (cache / f"{key}.wav").is_file() and (cache / f"{key}.json").is_file():
        meta = json.loads((cache / f"{key}.json").read_text())
        shutil.copyfile(cache / f"{key}.wav", target)
        os.utime(cache / f"{key}.wav")  # mais recente no cache
        logging.info("Cadeia já renderizada antes: reaproveitando (%s)", key)
        res = ChainResult(y=None, **meta)
        return spec, res, target, time.time() - t0, stages

    x, sr = read_wav_mono(args.input)
    if sr != SAMPLE_RATE:
        sys.exit(f"Sinal de treino precisa ser {SAMPLE_RATE} Hz (recebido {sr})")
    res = render_chain(x, stages)
    write_wav_24(target, res.y)
    logging.info("Cadeia renderizada em %.1fs -> %s", time.time() - t0, target)
    if cache:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            tmp = cache / f"{key}.tmp"
            shutil.copyfile(target, tmp)
            tmp.replace(cache / f"{key}.wav")
            (cache / f"{key}.json").write_text(json.dumps({
                "warnings": res.warnings, "clip_gain": res.clip_gain,
                "input_level_dbu": res.input_level_dbu, "stages_metadata": res.stages_metadata,
            }))
            _cache_prune(cache)
        except OSError as e:  # cache é só atalho: falhar não pode quebrar o job
            logging.warning("Sem cache de render: %s", e)
    return spec, res, target, time.time() - t0, stages


def _gear_type(stages_meta) -> str:
    """Tipo de equipamento do resultado (campo gear_type do .nam)."""
    flat = []
    for s in stages_meta:
        if s.get("type") == "split":
            for b in s["branches"]:
                flat.extend(b["stages"])
        else:
            flat.append(s)
    gears = {s.get("gear_type") for s in flat if s["type"] == "nam"}
    if any(s["type"] == "ir" for s in flat) or gears & {"amp_cab", "amp_pedal_cab"}:
        return "amp_cab"
    if gears & {"amp", "preamp", "pedal_amp", "studio"} or not gears or None in gears:
        return "amp"
    return "pedal"


def cmd_render(args) -> int:
    _render(args)
    return 0


def cmd_run(args) -> int:
    from .train import ESR_FAIL, esr_verdict, train_chain_model

    spec, res, target, t_render, stages = _render(args)
    arch = args.arch or spec.get("arch", "a2")
    name = spec.get("name") or Path(args.chain).stem
    metadata = {
        "name": name,
        "modeled_by": "NAM Chain Capture",
        "gear_type": _gear_type(res.stages_metadata),
        "chain_capture": {
            "stages": res.stages_metadata,
            "clip_gain": res.clip_gain,
            "warnings": res.warnings,
        },
    }
    if res.input_level_dbu is not None:
        metadata["input_level_dbu"] = res.input_level_dbu

    tr = train_chain_model(
        input_wav=args.input,
        output_wav=target,
        outdir=args.out,
        arch=arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        basename=args.basename,
        metadata=metadata,
        # Treino real mínimo (passa por checkpoints/export como um job de verdade).
        trainer_kwargs={"limit_train_batches": 2, "limit_val_batches": 1} if args.smoke else None,
        skip_input_check=args.skip_input_check,
        progress_path=Path(args.out) / "progress.json",
        init_from=None if args.no_warm_start else warm_start_candidates(stages),
        auto_stop=not args.no_auto_stop,
    )
    report = {
        "name": name,
        "arch": arch,
        "esr": tr.esr,
        "esr_by_size": tr.esr_by_size,
        "esr_verdict": esr_verdict(tr.esr),
        "epochs_run": tr.epochs_run,
        "epochs_max": args.epochs,
        "warm_start": tr.warm_start,
        "stop_reason": tr.stop_reason,
        "render_seconds": round(t_render, 2),
        "train_seconds": round(tr.seconds, 2),
        "gpu_cost_usd": (
            round(tr.seconds / 3600 * args.gpu_usd_per_hour, 4)
            if args.gpu_usd_per_hour
            else None
        ),
        "nam": str(tr.nam_path),
        "nam_a1": str(tr.legacy_nam_path) if tr.legacy_nam_path else None,
        "warnings": res.warnings,
        "stages": res.stages_metadata,
    }
    Path(args.out, "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if tr.esr > ESR_FAIL and not args.smoke:
        print(
            f"\nFALHOU: ESR {tr.esr:.4f} > {ESR_FAIL}. Tente mais épocas, uma arch "
            "maior, ou reduza o ganho/saturação da cadeia.",
            file=sys.stderr,
        )
        return 2
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="chaincap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("render", cmd_render), ("run", cmd_run)):
        s = sub.add_parser(name)
        s.add_argument("chain", help="chain.json")
        s.add_argument("--input", required=True, help="v3_0_0.wav oficial do NAM")
        s.add_argument("--out", required=True, help="pasta de saída")
        s.add_argument("--cache-dir", default=None,
                       help="pasta para reaproveitar renders da mesma cadeia")
        s.set_defaults(fn=fn)
        if name == "run":
            s.add_argument("--arch", choices=ARCHITECTURES)
            s.add_argument("--epochs", type=int, default=100)
            s.add_argument("--batch-size", type=int, default=16)
            s.add_argument("--basename", default="chain")
            s.add_argument("--gpu-usd-per-hour", type=float, default=None,
                           help="preço da GPU, para estimar custo do job")
            s.add_argument("--no-warm-start", action="store_true",
                           help="treina do zero, sem partir de uma captura da cadeia")
            s.add_argument("--no-auto-stop", action="store_true",
                           help="roda todas as épocas, mesmo depois de estabilizar")
            s.add_argument("--smoke", action="store_true",
                           help="treino de 2 passos (use com --epochs 1) só para testar o pipeline")
            s.add_argument("--skip-input-check", action="store_true",
                           help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
