"""Optional, persistent similarity search built from BitBIRCH clusters.

Install with ``pip install 'bblean[search]'``. Importing this module does not
import AWS clients or start a compiler. Fingerprints use NumPy's big bit order.
"""

from .index import SearchIndex
from .types import SearchResult, SearchStats

__all__ = ["SearchIndex", "SearchResult", "SearchStats"]
