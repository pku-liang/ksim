set(LEMON_ENABLE_GLPK NO CACHE STRING "Enable GLPK solver backend.")
set(LEMON_ENABLE_ILOG NO CACHE STRING "Enable ILOG (CPLEX) solver backend.")
set(LEMON_ENABLE_COIN NO CACHE STRING "Enable COIN solver backend.")
set(LEMON_ENABLE_SOPLEX NO CACHE STRING "Enable SoPlex solver backend.")

add_subdirectory(lemon-1.3.1 EXCLUDE_FROM_ALL)
target_compile_options(lemon PUBLIC -Wno-suggest-override)
target_include_directories(lemon PUBLIC lemon-1.3.1 ${CMAKE_CURRENT_BINARY_DIR}/lemon-1.3.1)