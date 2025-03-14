// REQUIRES: opencl-aot, cpu
// UNSUPPORTED: windows

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// CPU AOT targets host isa, so we compile on the run system instead.
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %{mathflags} %S/cmath_test.cpp -o %t.cmath.out
// RUN: %{run} %t.cmath.out

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %{mathflags} %S/cmath_fp64_test.cpp -o %t.cmath.fp64.out
// RUN: %{run} %t.cmath.fp64.out

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %{mathflags} %S/std_complex_math_test.cpp -o %t.complex.out
// RUN: %{run} %t.complex.out

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %{mathflags} %S/std_complex_math_fp64_test.cpp -o %t.complex.fp64.out
// RUN: %{run} %t.complex.fp64.out
