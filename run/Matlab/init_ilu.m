% Initialize block ILU0 method for use as a preconditioner
% Based on MFEM's BlockILU which is in linalg/solvers.cpp
function precond = init_ilu(diagA, A, bl_elems)
  if isa(A, 'function_handle')
    error("Unable to do block ILU on matrix-free operator!");
  end
  nlocal = size(diagA,1);
  nt = size(diagA,3);
  if nargin < 3; bl_elems = (1:nt).'; end;

  global t2t;
  t2t_local = convert_t2t(t2t, bl_elems);
  nf = size(t2t_local,1);
  nt_local = size(t2t_local,2);
  
  Dinvs = cell(nt_local,1); % used for storing LU factorizations of block diagonal
  b = create_blocks(nlocal, nt_local);
  
  AILU = A;
  for it = 1:nt_local
    i1 = b(1,it); in = b(2,it);
    
    % Find all nonzeros to the left of diagonal in row i
    for kk = 1:nf
      kt = t2t_local(kk,it);
      if kt > 0 && kt < it
	k1 = b(1,kt); kn = b(2,kt);
	% Right solve A_ik = A_ik * A_kk^{-1}
	AILU(i1:in,k1:kn) = AILU(i1:in,k1:kn)/Dinvs{kt};
	
	% Modify everything to the right of k in row i
	for jj = 1:nf
	  jt = t2t_local(jj,it);
	  if jt > 0 && jt > kt
	    j1 = b(1,jt); jn = b(2,jt);

	    AILU(i1:in,j1:jn) = AILU(i1:in,j1:jn) - AILU(i1:in,k1:kn)*AILU(k1:kn,j1:jn);
	  end
	end
        % Including diagonal (which is also to the right of k in row i)
	AILU(i1:in,i1:in) = AILU(i1:in,i1:in) - AILU(i1:in,k1:kn)*AILU(k1:kn,i1:in);
      end
    end
    
    % Compute new decomposition for use in all following operations
    Dinvs{it} = decomposition(AILU(i1:in,i1:in), 'lu');
  end
  
  % U\L\rhs
  precond = @(rhs) evaluate_ilu(Dinvs, AILU, t2t_local, b, rhs);
end

% Evaluate U\(L\rhs) stored in AILU
function x = evaluate_ilu(Dinvs, AILU, t2t_local, b, rhs)
  nt_local = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);
  nf = size(t2t_local,1);

  rhs_shape = reshape(rhs, nlocal,nt_local);
  y_shape = zeros(nlocal, nt_local);
  % Forward solve Ly = rhs where L has implicit identity on diagonal
  for it = 1:nt_local
    i1 = b(1,it); in = b(2,it);
    y_shape(:,it) = rhs_shape(:,it);
    for jj = 1:nf
      jt = t2t_local(jj,it);
      if jt > 0 && jt < it
	j1 = b(1,jt); jn = b(2,jt);
	y_shape(:,it) = y_shape(:,it) - AILU(i1:in,j1:jn)*y_shape(:,jt);
      end
    end
  end
  
  x_shape = y_shape;
  % Backward substitution to solve Ux = y
  for it = nt_local:-1:1
    i1 = b(1,it); in = b(2,it);
    for jj = 1:nf
      jt = t2t_local(jj,it);
      if jt > 0 && jt > it
	j1 = b(1,jt); jn = b(2,jt);
	x_shape(:,it) = x_shape(:,it) - AILU(i1:in,j1:jn)*x_shape(:,jt);
      end
    end
    x_shape(:,it) = Dinvs{it}\x_shape(:,it);
  end
  x = reshape(x_shape, nlocal*nt_local, 1);
end

% Convert t2t mapping from global elements to local subset of elements defined by bl_elems
function t2t_local = convert_t2t(t2t, bl_elems)
  nt = size(t2t,2);
  % Initialize mapping from element number to boundary element numbering
  elem2bl = zeros(nt,1);
  elem2bl(bl_elems) = 1:length(bl_elems);
  t2t_local = t2t(:,bl_elems);
  % Note: Only modify non-boundary condition values
  t2t_local(t2t_local > 0) = elem2bl(t2t_local(t2t_local > 0));
end

% Offset array to return the rows of matrix corresponding to element it
function blocks = create_blocks(nlocal, nt)
  blocks = zeros(2,nt);
  for it = 1:nt
    [blocks(1,it),blocks(2,it)] = block(it, nlocal);
  end
end

% Helper function to return the rows of matrix corresponding to element it
function [i1,in] = block(it, nlocal)
  i1 = nlocal*(it-1)+1;
  in = nlocal*(it);
end

% Recreate the element to element array from the nnz of the matrix itself
% TODO: Extremely costly
function t2t = recreate_t2t(A, b, nlocal, nt)
  nf = 4; 
  t2t = zeros(nf,nt);
  for it = 1:nt
    i1 = b(1,it); in = b(2,it);
    iF = 0;
    for jt = 1:nt
      if jt == it
	continue;
      end
      j1 = b(1,jt); jn = b(2,jt);

      if nnz(A(i1:in, j1:jn)) > 0 % Sparse matrices do not like being formed like this
	iF = iF + 1;
	t2t(iF,it) = jt;
      end
    end
  end
end
