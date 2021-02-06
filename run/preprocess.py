#!/usr/bin/env python

# Use:
# python preprocess.py -n N -s "suffix" -os "outputSuffix" -d "directory" -od "outputDirectory"
import argparse
import dgpython as dg

parser = argparse.ArgumentParser(description='Preprocess DG mesh.')
parser.add_argument('-n', dest='n', type=int, required=True,
                    help='Number of mesh partitions')
parser.add_argument('-s', dest='suffix', default='', help='string to add to input filename before .h5')
parser.add_argument('-os', dest='outsuffix', default='', help='string to add to output filename before .h5')
parser.add_argument('-d', dest='meshdir', default='../meshes/', help='directory containing input meshes')
parser.add_argument('-od', dest='outdir', default='./partitioned/', help='directory to output .h5 files')
args = parser.parse_args()

n = args.n
suffix = args.suffix
meshdir = args.meshdir
outdir = args.outdir
if (args.outsuffix == ''):
    newSuffix = args.suffix
else:
    newSuffix = args.outsuffix


msh = dg.Mesh(meshdir + "naca" + suffix + ".h5")

msh.partition(n)
msh.writefile(outdir + 'naca' + newSuffix + '_part' + str(n) + '.h5')
