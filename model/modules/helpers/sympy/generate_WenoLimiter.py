
import transformations as tr
import re
import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter
from sympy.printing.precedence import PRECEDENCE


class MyCXX11Printer(CXX11CodePrinter):
  def __init__(self, *, result_name="rslt", vector_name="v", real_type="real", **settings):
    super().__init__(settings)
    self.result_name = result_name
    self.vector_name = vector_name
    self.real_type = real_type
    self._vector_re = re.compile(rf"^{re.escape(vector_name)}(\d+)$")

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
        f"{self.result_name}{i} = {self._print(e)};"
        for i, e in enumerate(expr)
      )
    return super().doprint(expr, assign_to=assign_to)


def my_cxxcode(expr, *, result_name="rslt", vector_name="v", real_type="real", **settings):
  return MyCXX11Printer(
    result_name=result_name,
    vector_name=vector_name,
    real_type=real_type,
    **settings,
  ).doprint(expr)

print("\n#pragma once")

Nmax = 9
prec = 17

print("\ntemplate <class real, int ord> struct WenoLimiter;")

for N in range(3,Nmax+1,2) :
  NL = (N+1)//2
  idl_L,TV,L,R = tr.gen_weno_sten_to_edges_idl_TV(N)
  TVcode  = my_cxxcode(TV,result_name='    real TV',vector_name='v',real_type='real')
  Lcode   = my_cxxcode(L ,result_name='    real L' ,vector_name='v',real_type='real')
  Rcode   = my_cxxcode(R ,result_name='    real R' ,vector_name='v',real_type='real')
  print(f"\n\ntemplate <class real> struct WenoLimiter<real,{N}> "+"{")
  print(f"  static KOKKOS_INLINE_FUNCTION void compute(SArray<real,{N}> v, real &L, real &R) "+"{")
  print(TVcode)
  for i in range(NL) : print(f"    TV{i} *= TV{i};")
  print("    // Left Edge")
  for i in range(NL) :
    print(f"    real w{i} = static_cast<real>({idl_L[i].n(prec)})/(TV{i}+1.e-10);")
  print(f"    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),",end="")
  for i in range(NL) : print(f"w{i}",end=" + " if i<NL-1 else ");\n")
  for i in range(NL) : print(f"    w{i} *= r_sm;")
  print(Lcode)
  print("    L = ",end="")
  for i in range(NL) : print(f"w{i}*L{i}",end=" + " if i<NL-1 else ";\n")
  print("    // Right Edge")
  for i in range(NL) :
    print(f"    w{i} = static_cast<real>({idl_L[NL-1-i].n(prec)})/(TV{i}+1.e-10);")
  print(f"    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),",end="")
  for i in range(NL) : print(f"w{i}",end=" + " if i<NL-1 else ");\n")
  for i in range(NL) : print(f"    w{i} *= r_sm;")
  print(Rcode)
  print("    R = ",end="")
  for i in range(NL) : print(f"w{i}*R{i}",end=" + " if i<NL-1 else ";\n")
  print("  }")
  print("};")

