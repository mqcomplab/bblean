import dataclasses


@dataclasses.dataclass(slots=True)
class BitBirchConfig:
    threshold: float = 0.65
    branching_factor: int = 254
    merge_criterion: str = "diameter"
    tolerance: float = 0.05
    features_num: int = 2048


DEFAULTS = BitBirchConfig()
