# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import TypeVar
import numpy as np
import numpy.typing as npt


NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]
NDArrayUInt = npt.NDArray[np.uint64]

T = TypeVar('T', float, NDArrayFloat)