
import transformations as tr
import re
import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter
from sympy.printing.precedence import PRECEDENCE


class MyCXX11Printer(CXX11CodePrinter):
  def __init__(self, *, result_name="rslt", vector_name="v", real_type="real",i0=0, **settings):
    super().__init__(settings)
    self.result_name = result_name
    self.vector_name = vector_name
    self.real_type = real_type
    self._vector_re = re.compile(rf"^{re.escape(vector_name)}(\d+)$")
    self.i0 = i0

  def _print_Pow(self, expr):
    base, exp = expr.as_base_exp()
    if exp == 2:
      b = self.parenthesize(base, PRECEDENCE["Mul"])
      return f"{b}*{b}"
    return super()._print_Pow(expr)

  def _print_Float(self, expr):
    return f"static_cast<{self.real_type}>({str(expr)})"

  def _print_Symbol(self, expr):
    name = expr.name
    m = self._vector_re.match(name)
    if m:
      return f"{self.vector_name}({m.group(1)})"
    return super()._print_Symbol(expr)

  def doprint(self, expr, assign_to=None):
    if isinstance(expr, list):
      return "\n".join(
        f"{self.result_name}{self.i0+i} = {self._print(e)};"
        for i, e in enumerate(expr)
      )
    return super().doprint(expr, assign_to=assign_to)


def my_cxxcode(expr, *, result_name="rslt", vector_name="v", real_type="real",i0=0, **settings):
  return MyCXX11Printer(
    result_name=result_name,
    vector_name=vector_name,
    real_type=real_type,
    i0=i0,
    **settings,
  ).doprint(expr)

print("\n#pragma once")

Nmax = 9
prec = 17

print("\nnamespace TransformMatrices {")

print("")
print("  /////////////////////////////////////")
print("  // get_gll_points and get_gll_weights")
print("  /////////////////////////////////////")
for N in range(1,Nmax+1) :
  if (N >= 2) :
    pts,wts = tr.gll_pts_wts(N)
    print(f"  template <class F> KOKKOS_INLINE_FUNCTION void get_gll_points(yakl::SArray<F,{N}> & p) "+"{")
    for i in range(N) :
      print(f"    p({i}) = static_cast<F>({pts[i].n(prec)});")
    print("  }")
    print(f"  template <class F> KOKKOS_INLINE_FUNCTION void get_gll_weights(yakl::SArray<F,{N}> & w) "+"{")
    for i in range(N) :
      print(f"    w({i}) = static_cast<F>({wts[i].n(prec)});")
    print("  }")

print("\n")
print("  ////////////////")
print("  // coefs_to_sten")
print("  ////////////////")
for N in range(1,Nmax+1) :
  if (N%2==1) :
    A = tr.gen_coefs_to_sten(N)
    print(f"  template <class F> KOKKOS_INLINE_FUNCTION void coefs_to_sten(yakl::SArray<F,{N},{N}> & m) "+"{")
    for j in range(N) :
      for i in range(N) :
        print(f"    m({j},{i}) = static_cast<F>({A[i,j].n(prec)});")
    print("  }")

print("\n")
print("  ////////////////")
print("  // sten_to_coefs")
print("  ////////////////")
for N in range(1,Nmax+1) :
  if (N%2==1) :
    A = tr.gen_sten_to_coefs(N)
    print(f"  template <class F> KOKKOS_INLINE_FUNCTION void sten_to_coefs(yakl::SArray<F,{N},{N}> & m) "+"{")
    for j in range(N) :
      for i in range(N) :
        print(f"    m({j},{i}) = static_cast<F>({A[i,j].n(prec)});")
    print("  }")

print("\n")
print("  ///////////////")
print("  // gll_to_coefs")
print("  ///////////////")
for N in range(1,Nmax+1) :
  if N >= 2 :
    A = tr.gen_gll_to_coefs(N)
    print(f"  template <class F> KOKKOS_INLINE_FUNCTION void gll_to_coefs(yakl::SArray<F,{N},{N}> & m) "+"{")
    for j in range(N) :
      for i in range(N) :
        print(f"    m({j},{i}) = static_cast<F>({A[i,j].n(prec)});")
    print("  }")

print("\n")
print("  ///////////////")
print("  // coefs_to_gll")
print("  ///////////////")
print(f"  template <class F> KOKKOS_INLINE_FUNCTION void coefs_to_gll(yakl::SArray<F,1,2> & m) "+"{")
for i in range(2) :
  print(f"    m(0,{i}) = static_cast<F>(1);")
print("  }")
for N in range(2,Nmax+1) :
  for NQ in range(2,N+1) :
    A = tr.gen_coefs_to_gll(N,NQ)
    print(f"  template <class F> KOKKOS_INLINE_FUNCTION void coefs_to_gll(yakl::SArray<F,{N},{NQ}> & m) "+"{")
    for j in range(N) :
      for i in range(NQ) :
        print(f"    m({j},{i}) = static_cast<F>({A[i,j].n(prec)});")
    print("  }")

print("\n")
print("  ////////////////")
print("  // sampL, sampR")
print("  ////////////////")
for N in range(1,Nmax+1,2) :
  s2g = tr.gen_coefs_to_gll(N,2)*tr.gen_sten_to_coefs(N)*tr.gen_coefs(N,'s')
  L = s2g[0,:].tolist()[0][0]
  R = s2g[1,:].tolist()[0][0]
  print(f"  template <class F> KOKKOS_INLINE_FUNCTION real sampL(yakl::SArray<F,{N}> const & s) "+"{")
  print("    return ",end="")
  print(my_cxxcode(L,result_name='',vector_name='s'),end=";\n")
  print("  }")
  print(f"  template <class F> KOKKOS_INLINE_FUNCTION real sampR(yakl::SArray<F,{N}> const & s) "+"{")
  print("    return ",end="")
  print(my_cxxcode(R,result_name='',vector_name='s'),end=";\n")
  print("  }")

print("}\n")

