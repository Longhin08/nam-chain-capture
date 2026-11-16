"""Compatibilidade com o pacote `neural-amp-modeler` em máquinas headless.

`nam.train.core` importa `tkinter` no topo do módulo só para mostrar um modal de
aviso na GUI. Em servidores/containers sem Tk isso quebra o import. Como nunca
usamos a GUI, registramos um módulo vazio quando o Tk não está disponível.

Importar este módulo ANTES de qualquer `import nam`.
"""

import sys as _sys
import types as _types

try:  # pragma: no cover - depende do ambiente
    import tkinter  # noqa: F401
except ImportError:  # pragma: no cover
    _sys.modules["tkinter"] = _types.ModuleType("tkinter")
