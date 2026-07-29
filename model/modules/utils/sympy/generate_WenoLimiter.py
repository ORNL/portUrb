
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

print("\ntemplate <class real, int ord> struct WenoLimiter;")

for N in range(3,Nmax+1,2) :
  NL = (N+1)//2
  idl_L,TV,L,R,coeflist,TVgen = tr.gen_weno(N)
  TVcode  = my_cxxcode(TV,result_name='    real TV',vector_name='v',real_type='real')
  Lcode   = my_cxxcode(L ,result_name='    real L' ,vector_name='v',real_type='real')
  Rcode   = my_cxxcode(R ,result_name='    real R' ,vector_name='v',real_type='real')
  print(f"\n\ntemplate <class real> struct WenoLimiter<real,{N}> "+"{")
  print(f"  static KOKKOS_INLINE_FUNCTION void value_based(SArray<real,{N}> const & v, real &L, real &R, bool immL, bool immR) "+"{")

  # print("    real mn = ",end="")
  # for i in range(N-1) : print(f"std::min(v({i}),",end="" if i<N-2 else f"v({i+1})")
  # for i in range(N-1) : print(")",end="" if i<N-2 else ";\n")
  # print("    real mx = ",end="")
  # for i in range(N-1) : print(f"std::max(v({i}),",end="" if i<N-2 else f"v({i+1})")
  # for i in range(N-1) : print(")",end="" if i<N-2 else ";\n")
  # for i in range(N) : print(f"    real v{i} = (v({i})-mn)/std::max(static_cast<real>(1.e-20),mx-mn);")
  print(TVcode)
  for i in range(NL) : print(f"    TV{i} = std::abs(TV{i});")
  print(f"    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-20),",end="")
  for i in range(NL) : print(f"TV{i}",end=" + " if i<NL-1 else ");\n")
  for i in range(NL) : print(f"    TV{i} *= r_sm;")

  if N > 3 :
    print(f"    if (TV0==0) TV0 = ",end="")
    for i in range(1,NL-1) : print(f"std::min(TV{i},",end="" if i<NL-2 else f"TV{i+1}")
    for i in range(NL-2) : print(")",end="" if i<NL-3 else ";\n")

    print(f"    if (TV{NL-1}==0) TV{NL-1} = ",end="")
    for i in range(NL-2) : print(f"std::min(TV{i},",end="" if i<NL-3 else f"TV{i+1}")
    for i in range(NL-2) : print(")",end="" if i<NL-3 else ";\n")

  print(f"    if (immL) TV{NL-1} = ",end="")
  for i in range(NL-1) : print(f"std::max(TV{i},",end="" if i<NL-2 else f"TV{i+1}")
  for i in range(NL-1) : print(")",end="" if i<NL-2 else ";\n")

  print(f"    if (immR) TV0 = ",end="")
  for i in range(NL-1) : print(f"std::max(TV{i},",end="" if i<NL-2 else f"TV{i+1}")
  for i in range(NL-1) : print(")",end="" if i<NL-2 else ";\n")

  for i in range(NL) : print(f"    TV{i} *= TV{i};")

  print("    // Left Edge")
  for i in range(NL) :
    print(f"    real w{i} = static_cast<real>({idl_L[i].n(prec)})/(TV{i}+1.e-20);")
  print(f"    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-20),",end="")
  for i in range(NL) : print(f"w{i}",end=" + " if i<NL-1 else ");\n")
  for i in range(NL) : print(f"    w{i} *= r_sm;")
  print(Lcode)
  print("    L = ",end="")
  for i in range(NL) : print(f"w{i}*L{i}",end=" + " if i<NL-1 else ";\n")

  print("    // Right Edge")
  for i in range(NL) :
    print(f"    w{i} = static_cast<real>({idl_L[NL-1-i].n(prec)})/(TV{i}+1.e-20);")
  print(f"    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-20),",end="")
  for i in range(NL) : print(f"w{i}",end=" + " if i<NL-1 else ");\n")
  for i in range(NL) : print(f"    w{i} *= r_sm;")
  print(Rcode)
  print("    R = ",end="")
  for i in range(NL) : print(f"w{i}*R{i}",end=" + " if i<NL-1 else ";\n")
  print("  }")

  print("};")



# print("\ntemplate <class real, int ord> struct WenoLimiterEdgeCentered;")
# 
# for N in range(4,11,2) :
#   assert N%2==0 , "Polynomial degree must be even"
#   NL = N//2
#   NP = N//2+1
#   hs = N//2
#   # idl_L,TV,L,R,coeflist,TVgen = tr.gen_weno(N)
#   idl,TV,C,coeflist,TVgen = tr.gen_weno_edge(N)
#   TVcode  = my_cxxcode(TV,result_name='    real TV',vector_name='v',real_type='real')
#   Ccode   = my_cxxcode(C ,result_name='    real C' ,vector_name='v',real_type='real')
#   print(f"\n\ntemplate <class real> struct WenoLimiterEdgeCentered<real,{N}> "+"{")
#   print(f"  static KOKKOS_INLINE_FUNCTION real value_based(SArray<real,{N}> const & v, bool immL, bool immR) "+"{")
# 
#   print(TVcode)
#   for i in range(NP) : print(f"    TV{i} = std::abs(TV{i});")
#   print(f"    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-20),",end="")
#   for i in range(NP) : print(f"TV{i}",end=" + " if i<NP-1 else ");\n")
#   for i in range(NP) : print(f"    TV{i} *= r_sm;")
# 
#   if N == 4 :
#     print(f"    if (v(0)==v(1)) TV0 = std::min(TV1,TV2);")
#     print(f"    if (v(3)==v(2)) TV2 = std::min(TV0,TV1);")
#   elif N == 6 :
#     print(f"    if (v(0)==v(1) && v(0)==v(2)) TV0 = std::min(std::min(TV1,TV2),TV3);")
#     print(f"    if (v(5)==v(4) && v(5)==v(3)) TV3 = std::min(std::min(TV0,TV1),TV2);")
#   elif N == 8 :
#     print(f"    if (v(0)==v(1) && v(0)==v(2) && v(0)==v(3)) TV0 = std::min(std::min(std::min(TV1,TV2),TV3),TV4);")
#     print(f"    if (v(7)==v(6) && v(7)==v(5) && v(7)==v(4)) TV4 = std::min(std::min(std::min(TV0,TV1),TV2),TV3);")
#   elif N == 10 :
#     print(f"    if (v(0)==v(1) && v(0)==v(2) && v(0)==v(3) && v(0)==v(4)) TV0 = std::min(std::min(std::min(std::min(TV1,TV2),TV3),TV4),TV5);")
#     print(f"    if (v(9)==v(8) && v(9)==v(7) && v(9)==v(6) && v(9)==v(5)) TV5 = std::min(std::min(std::min(std::min(TV0,TV1),TV2),TV3),TV4);")
# 
#   print(f"    if (immL) TV{NP-1} = ",end="")
#   for i in range(NP-1) : print(f"std::max(TV{i},",end="" if i<NP-2 else f"TV{i+1}")
#   for i in range(NP-1) : print(")",end="" if i<NP-2 else ";\n")
# 
#   print(f"    if (immR) TV0 = ",end="")
#   for i in range(NP-1) : print(f"std::max(TV{i},",end="" if i<NP-2 else f"TV{i+1}")
#   for i in range(NP-1) : print(")",end="" if i<NP-2 else ";\n")
# 
#   for i in range(NP) : print(f"    TV{i} *= TV{i};")
# 
#   print("    // Edge")
#   for i in range(NP) :
#     print(f"    real w{i} = static_cast<real>({idl[i].n(prec)})/(TV{i}+1.e-20);")
#   print(f"    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-20),",end="")
#   for i in range(NP) : print(f"w{i}",end=" + " if i<NP-1 else ");\n")
#   for i in range(NP) : print(f"    w{i} *= r_sm;")
#   print(Ccode)
#   print("    return ",end="")
#   for i in range(NP) : print(f"w{i}*C{i}",end=" + " if i<NP-1 else ";\n")
# 
#   print("  }")
# 
#   print("};")
