curr_dir = "~/scratch/2021/naca/meshes/scripts/";
msh_dir  = curr_dir+"../";

ax = [-0.02 1.02 -0.08 0.08];
tail_pt = [1.0 0.0];
format long
hLEs = 0.0099:0.000001:0.0101;
% Attempt all of the hLEs as the leading edge length for our mesh (and possibly trailing edge length)
for i = 1:numel(hLEs)
  hLE = hLEs(i);
  try
    fprintf('\nTrying hLE = %.6f...', hLE);
%   msh = rungmsh2msh(msh_dir+"naca_v2.geo", "-order 1", ...
%		      "-setnumber hTE "+string(hTE), "-setnumber hLE "+string(hTE));
    msh = rungmsh2msh(msh_dir+"naca_v2.geo", "-order 1", ...
		      "-setnumber hLE "+string(hLE), "-setnumber hwing 0.049");

    p = msh.p';
    t = double(msh.t') + 1;
    e = unique(boundedges(p,t,t_block));
    e = e(dcircle(p(e,:),.5,0,1)<0);
    boundary_pts = sum(ismember(t,e),2);
    %good_bnd_layer = all(ismember(boundary_pts, [0,2]));
    bad_elements = ~ismember(boundary_pts, [0,2]);
    bad = t(bad_elements,:).';
    % Mesh is only a success if boundary layer around airfoil can be refined because 2 sides of all quads are connected to wall
    good_bnd_layer = ~any(bad_elements);
    for el = bad
      pts = p(el,:).';
      acceptable = false;
      for pt = pts
	if sum(abs(tail_pt - pt.')) < 1e-10
	  % Or if any problem elements are connected to tail pt
	  acceptable = true;
	  break
	end
      end
      good_bnd_layer = good_bnd_layer || acceptable;
      if ~acceptable
	fprintf("Bad element: \n")
	disp(p(el,:))
	good_bnd_layer = false;
	break
      end
    end
   
    if good_bnd_layer
      fprintf(' success! min(q) = %.3f\n', min(quadqual(p,t)));
      dgmeshplot(msh), axis(ax), drawnow
      break
    end
  catch
  end
end
