"""NAM Chain Capture — PoC (Épico 0).

Encadeia N capturas `.nam` + IRs, renderiza o sinal de treino padrão do NAM pela
cadeia e treina uma única captura nova que reproduz a cadeia inteira.
"""

from . import _compat  # noqa: F401  (precisa vir antes de qualquer import de nam)
from .chain import Stage, ChainResult, load_nam_model, load_ir, render_chain
from .archs import ARCHITECTURES, wavenet_config

__all__ = [
    "Stage",
    "ChainResult",
    "load_nam_model",
    "load_ir",
    "render_chain",
    "ARCHITECTURES",
    "wavenet_config",
]
