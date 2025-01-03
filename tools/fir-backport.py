#!/bin/bash

import re
import sys
from argparse import ArgumentParser
import re

parser = ArgumentParser('fir-backport.py')
parser.add_argument('input', help='input firrtl file', default='-')
parser.add_argument('-o', '--output', help='output firrtl file', default='-')
parser.add_argument('-v', '--verbose', help='verbose modification', action='store_true')
args = parser.parse_args()
verbose = args.verbose
fin = open(args.input) if args.input != '-' else sys.stdin
fout = open(args.output, 'w') if args.output != '-' else sys.stdout
ferr = sys.stderr

data = fin.read()
def trans_module(m: re.Match):
    if verbose:
        ferr.write(f'Modify {m.group(0)} => module\n')
    return 'module'
data = re.sub(r'public module', trans_module, data)
def trans_numlit(m: re.Match):
    w = m.group(2)
    tpe = m.group(4).lower()
    if tpe == 'b': base = 2
    if tpe == 'o': base = 8
    if tpe == 'x': base = 16
    v = int(m.group(5), base=base)
    res = f'{m.group(1)}Int<{w}>({m.group(3)}{v})'
    if verbose:
        ferr.write(f'Modify {m.group(0)} => {res}\n')
    return res
data = re.sub(r'([U|S])Int<(\d+)>\(([+-]?)0([bxoBXO])([01]+)\)', trans_numlit, data)
fout.write(data)
fin.close()
fout.close()
