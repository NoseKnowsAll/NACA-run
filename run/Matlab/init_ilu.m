% Initialize block ILU0 method for use as a preconditioner
% Based on MFEM's BlockILU which is in linalg/solvers.cpp
function precond = init_ilu(diagA, A)
  if isa(A, 'function_handle')
    error("Unable to do block ILU on matrix-free operator!");
  end
  
  nlocal = size(diagA,1);
  nt = size(diagA,3);

  % LU Factorization of block diagonal
  
  Dinvs = cell(nt,1);
  % TODO: Not sure if this is ever directly used...
  %for it = 1:nt
  %  Dinvs{it} = decomposition(diagA(:,:,it), 'lu');
  %end

  t2t = recreate_t2t(A, nlocal, nt);
  nf = size(t2t,1);
  
  AILU = A;
  for it = 1:nt
    [i1,in] = block(it,nlocal);
    
    % Find all nonzeros to the left of diagonal in row i
    for kt = t2t(:,it)
      if kt < it
	[k1,kn] = block(kt, nlocal);
	% Right solve A_ik = A_ik * A_kk^{-1}
	AILU(i1:in,k1:kn) = AILU(i1:in,k1:kn)/Dinvs{ik};
	
	% Modify everything to the right of k in row i
	for jt = t2t(:,it)
	  if jt > kt
	    [j1,jn] = block(jt, nlocal);

	    AILU(i1:in,j1:jn) = AILU(i1:in,j1:jn) - AILU(i1:in,k1:kn)*AILU(k1:kn,j1:jn);
	  end
	end
        % Including diagonal (which is also to the right of k in row i)
	AILU(i1:in,i1:in) = AILU(i1:in,i1:in) - AILU(i1:in,k1:kn)*AILU(k1:kn,i1:in);
	% Compute new decomposition for use in all following operations
	Dinvs{it} = decomposition(AILU(i1:in,i1:in), 'lu');
      end
    end
  end
  
  % U\L\rhs
  precond = @(rhs) evaluate_ilu(Dinvs, AILU, rhs);
end

% Evaluate U\(L\rhs) stored in AILU
function x = evaluate_ilu(Dinvs, AILU, t2t, rhs)
  nt = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);

  rhs_shape = reshape(rhs, nlocal,nt);
  y_shape = zeros(nlocal, nt);
  % Forward solve Ly = rhs where L has implicit identity on diagonal
  for it = 1:nt
    [i1,in] = block(it, nlocal);
    y_shape(:,it) = rhs_shape(:,it);
    for jt = t2t(:,it)
      if jt < it
	[j1,jn] = block(jt, nlocal);
	y_shape(:,it) = y_shape(:,it) - AILU(i1:in,j1:jn)*y_shape(:,jt);
      end
    end
  end
  
  x_shape = y_shape;
  % Backward substitution to solve Ux = y
  for it = nt:-1:1
    [i1,in] = block(it, nlocal);
    for jt = t2t(:,it)
      if jt > it
	[j1,jn] = block(jt, nlocal);
	x_shape(:,it) = x_shape(:,it) - AILU(i1:in,j1:jn)*x_shape(:,jt);
      end
    end
    x_shape(:,it) = Dinvs{it}\x_shape(:,it);
  end
  x = reshape(x_shape, nlocal*nt, 1);
end

% Recreate the element to element array from the nnz of the matrix itself
function t2t = recreate_t2t(A, nlocal, nt)
  nf = 4; % TODO: Figure this out programmatically
  t2t = zeros(nf,nt);
  for it = 1:nt
    [i1,in] = block(it, nlocal);
    iF = 0;
    for jt = 1:nt
      if jt == it
	continue;
      end
      [j1, jn] = block(jt, nlocal);
      A_ij = A(i1:in, j1:jn);

      if nnz(A_ij) > 0
	iF = iF + 1;
	t2t(iF,it) = jt;
      end
    end
  end
end

% Helper function to return the rows of matrix corresponding to element it
function [i1,in] = block(it, nlocal)
  i1 = nlocal*(it-1)+1;
  in = nlocal*(it);
end
