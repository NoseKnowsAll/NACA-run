#!/usr/bin/env python

# Usage: python mkanim.py start stop [movie=False] [images=True]

# Run in parallel with
# parallel --lb -j np -- < inputFile

import dgpython as dg
import numpy as np
import sys, subprocess
from pathlib import Path

p = 3
r = 12
msh_name = "naca_v2_p"+str(p)+"_r"+str(r)
naca_dir = "/scratch/mfranco/2021/naca/"
res_dir = naca_dir+"run/results/"+msh_name+"/"
msh_dir = naca_dir+"meshes/"
plt = dg.import_plt()
quant = "r"
msh = dg.Mesh(msh_dir+msh_name+".h5")
movie_name = res_dir+"mov/"+msh_name+quant+".mp4"

a = (-0.4,2.0,-1.2,1.2)
img = None

# Reads and returns the solution from a predefined file location
# i = time step value
def read_u(i):
  return msh.readsolution(res_dir+"snaps/sol%05d.dat" % (i))

# Return the filename corresponding to the image at timestep i
def img_filename(i):
  return res_dir+"img/"+quant+str(i).zfill(5)+".png"

# Actually creates the image of Mach value of the solution
def init():
  u = read_u(1)
  plt.gcf().set_size_inches(6,6)

  cmap = plt.cm.get_cmap("jet", 4096)
  cmap._init()
  cmap._lut[0,0:3] = 224./255

  global img
  img = plt.dgplot(msh, u, quant, eqn="ns")
  img.set_cmap("jet")
  plt.axis("equal")
  plt.axis(a)
  min_quant = 0
  max_quant = 2
  plt.clim(min_quant, max_quant)

# Saves the image to file
def plotit(i):
  u = read_u(i)
  img.set_data(u.copy("F"),quant,eqn="ns")
  plt.savefig(img_filename(i), dpi=300)
  print("Wrote frame "+str(i))

# index range with a given step
def irange(start, stop, step):
  return range(start, (stop + 1) if step >= 0 else (stop - 1), step)

# Converts string variables to boolean Python type
def str2bool(v):
  return str(v).lower() in ("yes", "true", "t", "1")

# Create list of filenames for imagees
# quant = quantity of interest
# frames = index range of filenames to movify
def getImgFilenameList(quant, frames):
    fnlist = []
    for i in frames:
        fn = res_dir+"img/"+quant+str(i).zfill(5)+".png"
        fnlist.append(fn)
    return fnlist

# Makes a movie from a list of image files
def makeMovie(imglist, moviename, crf=23, fps_out=24, fps_in=None):
  # Create temporary directory
  tmpdir = "tmpdir_for_im2mp4_deleteme"
  subprocess.call("mkdir {0:s}".format(tmpdir), shell=True)
  
  # Copy png into temporary directory
  for imgName in imglist:
    subprocess.call("cp {0:s} {1:s}/".format(imgName,tmpdir),
                    shell=True)
    
  # Call img2mp4
  exec_str = "img2mp4 {0:s} 0 {1:d} png {2:d}"
  if fps_in is not None: exec_str = exec_str + " " + str(fps_in)
  subprocess.call(exec_str.format(tmpdir, crf, fps_out), shell=True)
  
  # Move mp4 file
  subprocess.call("mv {0:s}.mp4 {1:s}".format(tmpdir, moviename), shell=True)
  
  # Delete temporary directory
  subprocess.call("rm {0:s}/*png".format(tmpdir), shell=True)
  subprocess.call("rmdir {0:s}".format(tmpdir), shell=True)



## START OF SCRIPT
argc = len(sys.argv)

# Defaults
movie = False
images = True

if argc > 1:
  start = sys.argv[1]
  stop = sys.argv[2]
else:
  print("ERROR: NEED TO SPECIFY START/END FRAMES")
  sys.exit()
if argc > 3:
  movie = str2bool(sys.argv[3])
if argc > 4:
  images = str2bool(sys.argv[4])

print("movie =", movie, "images =", images)

snapstep = 500
frames = irange(int(start), int(stop), snapstep)

if images is True:
  print("Plotting frames", frames)
  
  print("Initializing")
  init()
  
  print("Beginning to plot...")
  Path(res_dir+"img/").mkdir(parents=True, exist_ok=True)
  for i in frames:
    plotit(i)

if movie is True:
  print("Making movie...")
  Path(res_dir+"mov/").mkdir(parents=True, exist_ok=True)
  imgList = getImgFilenameList(quant,frames)
  makeMovie(imgList, movie_name)

