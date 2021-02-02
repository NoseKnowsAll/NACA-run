#!/usr/bin/env python

# Use:
# python preprocess.py -n N -s "suffix" -ns "newSuffix" -d "directory"
import argparse
import dgpython as dg

parser = argparse.ArgumentParser(description='Preprocess DG mesh.')
parser.add_argument('-n', dest='n', type=int, required=True,
                    help='Number of mesh partitions')
parser.add_argument('-s', dest='suffix', default='', help='string to add to input filename before .h5')
parser.add_argument('-ns', dest='newSuffix', default='', help='string to add to output filename before .h5')
parser.add_argument('-d', dest='meshdir', default='./partitioned/', help='directory containing meshes')
args = parser.parse_args()

n = args.n
suffix = args.suffix
meshdir = args.meshdir
if (args.newSuffix == ''):
    newSuffix = args.suffix
else:
    newSuffix = args.newSuffix


msh = dg.Mesh(meshdir + "naca" + suffix + ".h5")

msh.partition(n)
msh.writefile(meshdir + 'naca' + newSuffix + '_part' + str(n) + '.h5')
