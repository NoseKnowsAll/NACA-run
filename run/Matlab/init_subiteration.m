% Creates a subiteration preconditioner to solve A\b
function precond = init_subiteration(A, diagA, bl_elems, b, nsubiter)
  nt = size(diagA,3);
  nlocal = size(diagA,2);
  Dinvs = cell(nt,1);
  for it = 1:nt
    Dinvs{it} = decomposition(diagA(:,:,it), 'lu');
  end

  [A_bl, diagA_bl] = extract_suboperator(A, diagA, bl_elems);
  b_bl = extract_subvector(b, nt, nlocal, bl_elems);
  precond_bl = init_jacobi(A_bl, diagA_bl, b_bl);
  precond = @(x) evaluate_subiteration(A, A_bl, Dinvs, precond_bl, bl_elems, b, b_bl, nsubiter, x);
end

% Subiteration preconditioner is one application of P\x.
% First apply block diagonal inverse to entire region. x2 = D\x
% Then consider just the boundary layer elements specified by bl_elems.
% Compute boundary layer residual r_{bl} = b_{bl}-A_{bl}*x2_{bl}
% Do inner GMRES iteration to solve A_{bl}*x = r_{bl} with initial guess x2_{bl}
% Update y according to correction: y = x2 + x3
% Note: Only valid as a preconditioner within a broader FGMRES iteration
function y = evaluate_subiteration(A, A_bl, Dinvs, precond_bl, bl_elems, b, b_bl, nsubiter, x)

  nt = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);
  
  x2 = apply_global_preconditioner(Dinvs, x);
  if nsubiter <= 0
    y = x2;
    return;
  end

  r = A(x2) - b;
  r_bl = extract_subvector(r, nt, nlocal, bl_elems);
  r_global = pad_subvector(r_bl, nt, nlocal, bl_elems);
  r_nonbl = r - r_global; % r_nonbl contains residual away from region of interest
  b_global = pad_subvector(b_bl, nt, nlocal, bl_elems);
  b_nonbl = b - b_global; % b_nonbl contains RHS away from region of interest
  x2_bl = extract_subvector(x2, nt, nlocal, bl_elems);

  % TODO: It's not quite clear what this subiteration relative tolerance should be
  subtol = (norm(r_nonbl)/norm(b_nonbl)) / (norm(r_bl)/norm(b_bl));
  [y_bl, subiter, residuals] = static_gmres(A_bl, r_bl, x2_bl, subtol, nsubiter, precond_bl, "right", false);
  fprintf("Subiteration took %d iterations to achieve ||res|| = %f\n", subiter, residuals(end));
  y = x2;
  if subiter > 0
    y = y - pad_subvector(y_bl, nt, nlocal, bl_elems);
  end
  
end

% Evaluates global preconditioner to x. In this case, just block diagonal inverse x = D\x
function y = apply_global_preconditioner(Dinvs, x)
  nt = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);
  
  y = x;
  y2 = reshape(y, nlocal, nt);
  for it = 1:nt
    y2(:,it) = Dinvs{it}\y2(:,it);
  end
end

% Extract portion of operator A corresponding to rows specified by bl_elems
function [A_bl, diagA_bl] = extract_suboperator(Atimes, diagA, bl_elems)
  A_bl = @(x_bl) evaluate_suboperator(Atimes, diagA, bl_elems, x_bl);
  diagA_bl = diagA(:,:,bl_elems);
end

% Evaluates y_bl = A_bl*x_bl, when x is a subvector
function y_bl = evaluate_suboperator(Atimes, diagA, bl_elems, x_bl)
  nt = size(diagA,3);
  nlocal = size(diagA,2);

  x = pad_subvector(x_bl, nt, nlocal, bl_elems);
  y = Atimes(x);
  y_bl = extract_subvector(y, nt, nlocal, bl_elems);
end

% Given a global vector x, return the portion of the vector corresponding
% to the elements specified by bl_elems array.
function x_bl = extract_subvector(x, nt, nlocal, bl_elems)
  x2 = reshape(x, nlocal, nt);
  x_bl2 = x2(:,bl_elems);
  x_bl = reshape(x_bl2, nlocal*length(bl_elems), 1);
end

% Given a subvector x_bl, return a padded, global vector where
% x=x_bl for all the elements specified by bl_elems, and x=pad_val for all others
function x = pad_subvector(x_bl, nt, nlocal, bl_elems, pad_val)
  if nargin < 5
    pad_val = 0;
  end

  x_bl_shape = reshape(x_bl, nlocal, length(bl_elems));
  if pad_val == 0
    x_shape = zeros(nlocal, nt);
  else
    x_shape = ones(nlocal,nt)*pad_val;
  end
  x_shape(:,bl_elems) = x_bl_shape(:,:);
  x = reshape(x_shape, nlocal*nt, 1);
end
