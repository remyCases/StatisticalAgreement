# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatiscalAgreement.

import argparse
import StatisticalAgreement as sa
from examples import examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", "-e", required=False, action='store_true')
    parser.add_argument("--test", "-t", required=False, action='store_true')
    parser.add_argument("--simulation","-s", required=False, nargs='*')

    args = parser.parse_args()

    if args.simulation:
        sa.mc_simulation(*args.simulation)

    if args.example:
        examples.main()