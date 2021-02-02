curr_dir = '~/scratch/2021/naca/meshes/scripts/';
msh_dir  = strcat(curr_dir,'../');
p = 3
refinements = [0, 2, 4, 8]

msh = rungmsh2msh(strcat(msh_dir,'naca_v1.geo'), strcat('-order ',int2str(p)));
mshp = mshchangep(msh, p);
for refine = refinements
  msh = qmshbndlayer(mshp, [1], refine);
  msh = mshreorder(msh, 'weight', [2,1]);

  h5fwritestruct(msh, strcat(msh_dir,'naca_v1_p',int2str(p),'_r',int2str(refine),'.h5'), dgfieldnames('msh'));
end

