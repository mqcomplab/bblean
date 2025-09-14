import numpy as np
from numpy.typing import NDArray, DTypeLike
from scipy.stats import truncnorm


def make_fake_fingerprints(
    num: int,
    n_features: int = 2048,
    pack: bool = False,
    seed: int | None = None,
    dtype: DTypeLike = np.uint8,
) -> NDArray[np.uint8]:
    # Generate "synthetic" fingerprints with a popcount distribution
    # similar to one in a real smiles database
    # Fps are guaranteed to *not* be all zeros or all ones
    if pack:
        if np.dtype(dtype) != np.dtype(np.uint8):
            raise ValueError("Only np.uint8 dtype is supported for packed input")
    loc = 750
    scale = 400
    bounds = (0, n_features)
    rng = np.random.default_rng(seed)
    safe_bounds = (bounds[0] + 1, bounds[1] - 1)
    a = (safe_bounds[0] - loc) / scale
    b = (safe_bounds[1] - loc) / scale
    popcounts_fake_float = truncnorm.rvs(
        a, b, loc=loc, scale=scale, size=num, random_state=rng
    )
    popcounts_fake = np.rint(popcounts_fake_float).astype(np.int64)
    zerocounts_fake = n_features - popcounts_fake
    repeats_fake = np.empty((num * 2), dtype=np.int64)
    repeats_fake[0::2] = popcounts_fake
    repeats_fake[1::2] = zerocounts_fake
    initial = np.tile(np.array([1, 0], np.uint8), num)
    expanded = np.repeat(initial, repeats=repeats_fake)
    fps_fake = rng.permuted(expanded.reshape(num, n_features), axis=-1)
    if pack:
        return np.packbits(fps_fake, axis=1)
    return fps_fake.astype(dtype, copy=False)
