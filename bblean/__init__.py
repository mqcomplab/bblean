r"""BitBIRCH-Lean, a high-throughput, memory-efficient implementation of BitBIRCH

BitBIRCH-Lean is designed for high-thorouput clustering of huge molecular
libraries (of up to hundreds of milliones of molecules).
"""

import bblean.plotting as plotting
import bblean.analysis as analysis
import bblean.fingerprints as fingerprints
import bblean.multiround as multiround
import bblean.utils as utils
import bblean.smiles as smiles
import bblean.bitbirch as bitbirch
from bblean.bitbirch import BitBirch, set_merge  # type: ignore
from bblean.fingerprints import pack_fingerprints, unpack_fingerprints

__all__ = [
    # Modules
    "bitbirch",
    "multiround",
    "plotting",
    "analysis",
    "smiles",
    "fingerprints",
    "utils",
    # Global namespace for convenience
    "BitBirch",
    "set_merge",
    "pack_fingerprints",
    "unpack_fingerprints",
]
