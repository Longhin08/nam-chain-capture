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
import json
import logging
import sys
import time
from pathlib import Path

from .archs import ARCHITECTURES
from .chain import parse_node, read_wav_mono, render_chain, write_wav_24, SAMPLE_RATE


def _load_chain(path: Path):
    spec = json.loads(path.read_text())
    stages = [parse_node(s, base_dir=path.parent) for s in spec["stages"]]
    return spec, stages


def _render(args) -> tuple:
    spec, stages = _load_chain(Path(args.chain))
    x, sr = read_wav_mono(args.input)
    if sr != SAMPLE_RATE:
        sys.exit(f"Sinal de treino precisa ser {SAMPLE_RATE} Hz (recebido {sr})")
    t0 = time.time()
    res = render_chain(x, stages)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "target.wav"
    write_wav_24(target, res.y)
    logging.info("Cadeia renderizada em %.1fs -> %s", time.time() - t0, target)
    return spec, res, target, time.time() - t0


def cmd_render(args) -> int:
    _render(args)
    return 0


def cmd_run(args) -> int:
    from .train import ESR_FAIL, esr_verdict, train_chain_model

    spec, res, target, t_render = _render(args)
    arch = args.arch or spec.get("arch", "standard")
    name = spec.get("name") or Path(args.chain).stem
    metadata = {
        "name": name,
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
    )
    report = {
        "name": name,
        "arch": arch,
        "esr": tr.esr,
        "esr_verdict": esr_verdict(tr.esr),
        "epochs_run": tr.epochs_run,
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
        s.set_defaults(fn=fn)
        if name == "run":
            s.add_argument("--arch", choices=ARCHITECTURES)
            s.add_argument("--epochs", type=int, default=100)
            s.add_argument("--batch-size", type=int, default=16)
            s.add_argument("--basename", default="chain")
            s.add_argument("--gpu-usd-per-hour", type=float, default=None,
                           help="preço da GPU, para estimar custo do job")
            s.add_argument("--smoke", action="store_true",
                           help="treino de 2 passos (use com --epochs 1) só para testar o pipeline")
            s.add_argument("--skip-input-check", action="store_true",
                           help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
