# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from StatisticalAgreement.core import _categorical_agreement
from StatisticalAgreement.core.classutils import Indices
from StatisticalAgreement.core.agreement import AgreementFunctor, AgreementIndex, agreement
from StatisticalAgreement.simulation.mc_simulation import mc_simulation

ccc = AgreementIndex(name=Indices.CCC)
assert isinstance(ccc, AgreementFunctor)

cp = AgreementIndex(name=Indices.CP)
assert isinstance(cp, AgreementFunctor)

tdi = AgreementIndex(name=Indices.TDI)
assert isinstance(tdi, AgreementFunctor)

msd = AgreementIndex(name=Indices.MSD)
assert isinstance(msd, AgreementFunctor)

kappa = AgreementIndex(name=Indices.KAPPA)
assert isinstance(kappa, AgreementFunctor)

get_contingency_table = _categorical_agreement.contingency
