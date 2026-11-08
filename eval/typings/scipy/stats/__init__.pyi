# Minimal stub for the part of scipy this package uses (scipy ships no type information).
import numpy as np
from numpy.typing import ArrayLike, NDArray

def rankdata(a: ArrayLike, method: str = "average") -> NDArray[np.float64]: ...
