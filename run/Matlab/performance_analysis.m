% Check performance of subiteration solver on NACA problem to analyze several different hyperparameters
function performance_analysis()

  % Initialization
  dts = [1e-4, 1e-3, 1e-2];
  p   = 3;
  global_precond_type = "jacobi";
  tol = 1e-8;
  subtol_fudges = [1e0 1e1 1e2];
  nsubiters = [10 20 300];

  performance = zeros(length(nsubiters), length(subtol_fudges), length(dts), 2);

  % Run tests 
  for idt = 1:length(dts);
    dt = dts(idt);

    for ist = 1:length(subtol_fudges)
      subtol_fudge = subtol_fudges(ist);

      for ins = 1:length(nsubiters);
	nsubiter = nsubiters(ins);
	
	naca_subiteration_driver(dt, tol, nsubiter, subtol_fudge, global_precond_type);
	
	global inner_iterations;
	global outer_iteration;
	performance(ins, ist, idt, 1) = inner_iterations;
	performance(ins, ist, idt, 2) = outer_iteration;
	fprintf("\n\n\n\n");
	
      end
    end
  end

  % Print results at the end
  for idt = 1:length(dts)
    dt = dts(idt);
    for ist = 1:length(subtol_fudges)
      subtol_fudge = subtol_fudges(ist);
      for ins = 1:length(nsubiters)
	nsubiter = nsubiters(ins);

	fprintf("@ dt=%.1e, subtol_factor=%.1e, nsubiter=%3d, (inner, outer) = (%d, %d)\n", dt, subtol_fudge, nsubiter, performance(ins,ist,idt,1), performance(ins,ist,idt,2));
      end
    end
  end
  disp(performance);
  save("../results/Matlab/performance.mat", "performance");
  
end
