function study_subiteration()

  dts = [1e-4, 1e-3, 1e-2];
  hs  = [1e-3, 1e-2, 1e-1];
  p   = 3;
  global_precond_type = "mass_inv";
  nsubiters = [500];

  for h = hs
    for dt = dts
      for nsubiter = nsubiters
	subiteration_driver(dt, nsubiter, global_precond_type, h, p);
	fprintf("\n\n\n\n");
      end
    end
  end
  
end
