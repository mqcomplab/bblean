r"""BitBIRCH-Lean, a high-throughput, memory-efficient implementation of BitBIRCH

BitBIRCH-Lean is designed for high-thorouput clustering of huge molecular
libraries (of up to hundreds of milliones of molecules).
"""

from bblean.smiles import load_smiles
from bblean.fingerprints import fps_from_smiles
from bblean.bitbirch import BitBirch, set_merge
from bblean.fingerprints import pack_fingerprints, unpack_fingerprints
from bblean._version import __version__

__all__ = [
    # Global namespace for convenience
    "BitBirch",
    "set_merge",
    "pack_fingerprints",
    "unpack_fingerprints",
    "load_smiles",
    "fps_from_smiles",
    "__version__",
]
