function extended_msh = extend_naca_bounding_box(msh, R)
  % extended_msh = extend_naca_bounding_box(msh, R)
  % Extends the specified mesh's bounding box from R = 5 to specified R
  % Resulting mesh will have bounding box [-0.5-R, 0.5+2*R] x [-R, R]
  if nargin < 2
    R = 10;
  end
  if nargin < 1
    msh = h5freadstruct("../naca_v2_p3_r0.h5");
  end
  porder = msh.porder;
  R_orig = 5; % TODO: Determine from mesh itself?
  if R == R_orig
    extended_msh = msh;
    return;
  end
  assert(R > R_orig, "You must extend the original mesh, not shrink it!");

  % Create p,t by extending original mesh in all four directions
  p = msh.p';
  t = msh.t'+1; % Matlab is 1-indexed
  i_left  = x_lies(p, 0.5-R_orig);
  i_right = flip(x_lies(p, 0.5+2*R_orig));
  [p,t] = extend_in_direction(p, t, i_left,  [-1, 0], R - R_orig);
  [p,t] = extend_in_direction(p, t, i_right, [+1, 0], 2*(R - R_orig));
  
  i_top = y_lies(p, R_orig);
  i_bottom = flip(y_lies(p, -R_orig));
  [p,t] = extend_in_direction(p, t, i_top,     [0, +1], R - R_orig);
  [p,t] = extend_in_direction(p, t, i_bottom,  [0, -1], R - R_orig);

  % Create new mesh from now fully-defined p,t
  extended_msh = finalize_extended_mesh(msh, p, t, R);
  extended_msh = nodealloc(extended_msh, porder);
  extended_msh = mshchangep(extended_msh, porder);
  
end

% Extend the points specified by indices i_pts in the 2-vec direction a specified amount of times
% The direction vector also specifies the size of the new elements created
% (i.e. a direction = [-2, 0] will yield elements of hx = 2)
function [p,t] = extend_in_direction(p, t, i_pts, direction, amount)
  pts = p(i_pts,:);
  np = size(p,1);
  npslice = size(pts,1);
  pts_to_add = [];
  ts_to_add = zeros(npslice-1,amount,4);

  % pts simply fill in grid based on direction vector and given points
  dir_vec = repmat(direction, npslice, 1);
  for dx = 1:amount
    pts = pts + dir_vec;
    pts_to_add = [pts_to_add; pts];
  end
  % First add ts that connect to previously defined boundary points and new points
  ts_to_add(:,1,1) = i_pts(1:npslice-1);
  ts_to_add(:,1,2) = i_pts(2:npslice);
  ts_to_add(:,1,3) = (1:npslice-1)+np;
  ts_to_add(:,1,4) = (2:npslice)+np;
  % Then add ts that only connect newly defined points
  for dx = 1:(amount-1)
    ts_to_add(:,dx+1,1) = (1:npslice-1)+((dx-1)*npslice+np);
    ts_to_add(:,dx+1,2) = (2:npslice)  +((dx-1)*npslice+np);
    ts_to_add(:,dx+1,3) = (1:npslice-1)+(dx*npslice+np);
    ts_to_add(:,dx+1,4) = (2:npslice)  +(dx*npslice+np);
  end
  ts_to_add = reshape(ts_to_add, [(npslice-1)*amount,4]);

  p = [p; pts_to_add];
  t = [t; ts_to_add];
end

% Return indices of all points that lie at x = x_val. Ensure they're sorted from smallest to largest y
function on_val = x_lies(pts, x_val)
  tol = 1e-6;
  on_val = find(abs(pts(:,1) - x_val) < tol);
  [~,permute] = sort(pts(on_val, 2));
  on_val = on_val(permute);
end
% Return indices of all points that lie at y = y_val. Ensure they're sorted from smallest x to largest x
function on_val = y_lies(pts, y_val)
  tol = 1e-6;
  on_val = find(abs(pts(:,2) - y_val) < tol);
  [~,permute] = sort(pts(on_val, 1));
  on_val = on_val(permute);
end

function msh = finalize_extended_mesh(old_msh, p, t, R)
  [p,t] = fixmesh(p,t);
  
  bnd2=sprintf('all((p(:,2) > %.1f-1e-6) | (p(:,2) < -%.1f+1e-6) | (p(:,1) > 0.5+2*%.1f-1e-6) | (p(:,1) < 0.5-%.1f+1e-6))',R,R,R,R);
  bndexpr = {'all(abs(p(:,2))<1) & all(abs(p(:,1))<2)', bnd2};
  nbnd = length(bndexpr);
  
  msh = ml2msh(p,t,bndexpr);
  msh = mshcurved(msh, "all");
  if isfield(old_msh,"bndnames") && length(old_msh.bndnames) == nbnd
    msh.bndnames = cell(1, nbnd);
    for i = 1:nbnd
      msh.bndnames{i} = old_msh.bndnames{i};
    end
  end
end
