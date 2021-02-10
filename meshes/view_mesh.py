#!/usr/bin/env python

# Use:
# py3dg view_mesh.py -m "meshname"
import argparse
import dgpython as dg

parser = argparse.ArgumentParser(description='View mesh')
parser.add_argument('-m', dest='meshname', default='naca_v2_p3_r0.h5', help='meshname to view')
args = parser.parse_args()

plt = dg.import_plt()
msh = dg.Mesh(args.meshname)
dg.meshplot_curved(msh)
plt.show()
