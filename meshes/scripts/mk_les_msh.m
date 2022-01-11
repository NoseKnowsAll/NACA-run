% Should we store boundary layer elements
store_bl = 1;

% Run script to create .msh/bl.mat file
curr_dir = "~/scratch/2021/naca/meshes/scripts/";
msh_dir  = curr_dir+"../";
p = 3;
refine = 3;

msh = rungmsh2msh(msh_dir+"les.geo","-order "+string(p));
mshp = mshchangep(msh, p);

[msh, bnd_elems] = qmshbndlayer(mshp, [1], refine);
[msh, perm] = mshreorder(msh, 'weight', [2,1]);

if length(bnd_elems) > 0 && store_bl
  % Reorder boundary elements according to same permutation as overall elements got reordered
    %[~,inv_perm] = sort(perm); % Equivalent to below, but less efficient
    inv_perm(perm) = 1:numel(perm);
    bnd_elems = int32(inv_perm(bnd_elems));
    
    % Save boundary layer elements as .mat file
    fid = fopen(msh_dir+"les_p"+string(p)+"bl.mat", "w");
    fwritearray(fid, bnd_elems);
    fclose(fid);
end

h5fwritestruct(msh, msh_dir+"les_p"+string(p)+".h5", dgfieldnames('msh'));
