% Two values computed from find_hs script to ensure successful boundary layer
hwing = 0.049;
hLE = 0.0099;

% Should we store boundary layer elements
store_bl = 1

% Run script to create 
curr_dir = "~/scratch/2021/naca/meshes/scripts/";
msh_dir  = curr_dir+"../";
p = 3;
refinements = [0, 2, 4, 8, 12]
msh = rungmsh2msh(msh_dir+"naca_v2.geo", "-order "+string(p), "-setnumber hwing "+string(hwing), "-setnumber hLE "+string(hLE));
mshp = mshchangep(msh, p);
for refine = refinements
  [msh, bnd_elems] = qmshbndlayer(mshp, [1], refine);
  [msh,perm] = mshreorder(msh, 'weight', [2,1]);
  
  if length(bnd_elems) > 0 && store_bl
    % Reorder boundary elements according to same permutation as overall elements got reordered
    %[~,inv_perm] = sort(perm); % Equivalent to below, but less efficient
    inv_perm(perm) = 1:numel(perm);
    bnd_elems = inv_perm(bnd_elems);
    
    % Save boundary layer elements as .mat file
    fid = fopen(msh_dir+"naca_v2_p"+string(p)+"_r"+string(refine)+"bl.mat", "w");
    fprintf(fid, "%7d\n", bnd_elems);
    fclose(fid);
  end
  h5fwritestruct(msh, msh_dir+"naca_v2_p"+string(p)+"_r"+string(refine)+".h5", dgfieldnames('msh'));
end


