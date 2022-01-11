cmake_minimum_required(VERSION 3.5)

set(ROOT3DG $ENV{ROOT3DG})
if(NOT EXISTS ${ROOT3DG})
  message(FATAL_ERROR "ROOT3DG environment variable must be defined")
endif()

include(${ROOT3DG}/3dg.cmake)

dg_app_mpi(les2d les2d.cpp)
dg_app_mpi(timing2d timing2d.cpp)
dg_app_mpi(benchmark_dgops benchmark_dgops.cpp)
dg_app_mpi(eigenvalue eigenvalue.cpp)
dg_app_mpi(initMatlab initMatlab.cpp)
dg_app_mpi(sweep_dt sweep_dt.cpp)
dg_app_mpi(sweep_dt_bl sweep_dt_bl.cpp)
