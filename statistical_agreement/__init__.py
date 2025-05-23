# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from statistical_agreement.core import _categorical_agreement
from statistical_agreement.core.classutils import Indices
from statistical_agreement.core.agreement import AgreementFunctor, AgreementIndex
from statistical_agreement.core.agreement import agreement as agreement
from statistical_agreement.simulation.mc_simulation import mc_simulation as mc_simulation


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
