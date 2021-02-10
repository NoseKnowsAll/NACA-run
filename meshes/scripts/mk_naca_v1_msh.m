curr_dir = "~/scratch/2021/naca/meshes/scripts/";
msh_dir  = curr_dir+"../";
p = 3
refinements = [0, 2, 4, 8, 12]
msh = rungmsh2msh(msh_dir+"naca_v1.geo", "-order "+string(p));
mshp = mshchangep(msh, p);
for refine = refinements
  msh = qmshbndlayer(mshp, [1], refine);
  msh = mshreorder(msh, 'weight', [2,1]);

  h5fwritestruct(msh, msh_dir+"naca_v1_p"+string(p)+"_r"+string(refine)+".h5", dgfieldnames('msh'));
end

