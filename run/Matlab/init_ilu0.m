% Initialize block ILU0 method for use as a preconditioner
% Based on MFEM's BlockILU which is in linalg/solvers.cpp
function precond = init_ilu0(diagA, A)
  % LU Factorization of block diagonal
  nt = size(diagA,3);
  Dinvs = cell(nt,1);
  for it = 1:nt
    Dinvs{it} = decomposition(diagA(:,:,it), 'lu');
  end

  %DILU(:) = DA(:)
  %OILU(:) = OA(:) % Should not be needed
  %loop through all elements
  %  DILU(it) = DILU(it)^{-1}
  %  loop through all neighbors not yet visited
  %    OILU(them_for_me) = OA(them_for_me)*DILU(it)
  %    DILU(them) -= OILU(them_for_me)*OILU(me_for_them)

  % TODO: Make the below actually work
  for it = 1:nt
    for kt = 1:it
      if kt in neighbors(it)
	A(:,:,it,kt) = A(:,:,it,kt)/Dinvs{it};
	for jt = kt+1:nt
	  if jt in neighbors(it)
	    A(:,:,it,jt) = A(:,:,it,jt) - A(:,:,it,kt)*A(:,:,it,jt);
	    if jt == it
	      Dinvs{it} = decomposition(A(:,:,it,it), 'lu');
	    end
	  end
	end
      end
    end
  end
  
  % L/U
  precond = @(rhs) evaluate_ilu0(Dinvs, OA, rhs);
end

function x = evaluate_ilu0(Dinvs, OA, rhs)
  nt = size(Dinvs,1);
  nlocal = Dinvs{1}.MatrixSize(2);

end
