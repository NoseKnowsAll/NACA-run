function mk_naca_v2_R_msh()
  % Two values computed from find_hs script to ensure successful boundary layer
  hwing = 0.049;
  hLE = 0.0099;

  % Should we store boundary layer elements
  store_bl = 1;

  % Run script to create 
  curr_dir = "~/scratch/2021/naca/meshes/scripts/";
  msh_dir  = curr_dir+"../";
  p = 3;
  refinements = [0, 12]
  Rs = [5, 10, 20, 30, 40]
  msh_orig = rungmsh2msh(msh_dir+"naca_v2.geo", "-order "+string(p), "-setnumber hwing "+string(hwing), "-setnumber hLE "+string(hLE));
  msh_orig = mshchangep(msh_orig, p); %TODO: I do not think this is strictly necessary
  for R = Rs
    msh = extend_naca_bounding_box(msh_orig, R);
    msh = mshchangep(msh, p);

    for refine = refinements
      [msh, bnd_elems] = qmshbndlayer(msh, [1], refine);
      [msh, perm] = mshreorder(msh, 'weight', [2,1]);
    
      if length(bnd_elems) > 0 && store_bl
        % Reorder subregion elements according to same permutation because of mshreorder above
        %[~,inv_perm] = sort(perm); % Equivalent to below, but less efficient
	inv_perm(perm) = 1:numel(perm);
	bnd_elems = int32(inv_perm(bnd_elems));
	
        % Save boundary layer elements as .mat file
	fid = fopen(msh_dir+"naca_v2_p"+string(p)+"_r"+string(refine)+"_R"+string(R)+"bl.mat", "w");
	fwritearray(fid, bnd_elems);
	fclose(fid);
      end
      h5fwritestruct(msh, msh_dir+"naca_v2_p"+string(p)+"_r"+string(refine)+"_R"+string(R)+".h5", dgfieldnames('msh'));
    end
  end
end
