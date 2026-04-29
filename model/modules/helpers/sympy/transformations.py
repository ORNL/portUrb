
import sympy as sp

def gen_coefs(N,lab,i0=0) :
  return sp.Matrix(sp.symbols(f"{lab}{i0+0}:{i0+N}"))


def gen_poly(coefs) :
  x = sp.symbols('x')
  return sum([ coefs[i]*x**i for i in range(len(coefs)) ])


def gen_fv_stencil_constraints(p) :
  N  = sp.degree(p,gen=sp.symbols('x')) + 1  # Number DOFs
  assert N%2==1 , "Polynomial degree must be odd"
  hs = (N-1)//2  # "halo" size
  x  = sp.symbols('x')
  return sp.Matrix([ sp.integrate(p,(x,sp.Rational(2*i-1,2),sp.Rational(2*i+1,2))) for i in range(-hs,hs+1) ])


def gll_pts_wts(N) :
  assert N>=2,"gll_pts_wts only defined for N>=2"
  from sympy.integrals.quadrature import gauss_lobatto
  pts,wts = gauss_lobatto(N,17)
  return sp.Matrix([pts[i]/2 for i in range(N)]) , sp.Matrix([(wts[i])/2 for i in range(N)])


def gen_coefs_to_sten(N) :
  coefs  = gen_coefs(N,'a')
  p      = gen_poly(coefs)
  constr = gen_fv_stencil_constraints(p)
  return constr.jacobian(coefs)


def gen_sten_to_coefs(N) :
  return gen_coefs_to_sten(N).inv()


def gen_coefs_to_gll(N,NQ) :
  x       = sp.symbols('x')
  coefs   = gen_coefs(N,'a')
  p       = gen_poly(coefs)
  pts,wts = gll_pts_wts(NQ)
  constr  = sp.Matrix([ p.subs(x,pt) for pt in pts ])
  A       = constr.jacobian(coefs)
  return A


def gen_gll_to_coefs(N) :
  return gen_coefs_to_gll(N,N).inv()
  

def gen_weno(N) :
  assert N%2==1 , "Polynomial degree must be odd"
  NL = (N+1)//2
  hs = (N-1)//2  # "halo" size
  edges    = []
  TVlist   = []
  coeflist = []
  for ipL in range(NL) :
    coefs     = gen_coefs(NL,'a')
    p         = gen_poly(coefs)
    x         = sp.symbols('x')
    constr    = sp.Matrix([ sp.integrate(p,(x,sp.Rational(2*i-1,2),sp.Rational(2*i+1,2))) for i in range(-hs+ipL,-hs+ipL+NL) ])
    Ainv      = constr.jacobian(coefs).inv()
    vals      = gen_coefs(NL,'v',ipL)
    coefs     = Ainv * vals
    coeflist += [coefs.n(17).transpose().tolist()[0]]
    p         = gen_poly(coefs)
    edges    += [sp.Matrix([ p.subs(x,-sp.Rational(1,2)) , p.subs(x,sp.Rational(1,2)) ])]
    TV        = sum([ sp.integrate(sp.diff(p,x,i)**2,(x,-sp.Rational(1,2),sp.Rational(1,2))) for i in range(1,NL) ]).expand()
    TVlist   += [TV.n(17)]
  coefs   = gen_coefs(N,'a')
  p       = gen_poly(coefs)
  x       = sp.symbols('x')
  constr  = gen_fv_stencil_constraints(p)
  Ainv    = constr.jacobian(coefs).inv()
  vals    = gen_coefs(N,'v')
  coefs   = Ainv * vals
  p       = gen_poly(coefs)
  edges_h = sp.Matrix([ p.subs(x,-sp.Rational(1,2)) , p.subs(x,sp.Rational(1,2)) ])
  # Ideal weights at left edge
  idl     = gen_coefs(NL,'idl')
  A       = sp.Matrix([ sum([idl[i]*edges[i][0] for i in range(NL)]) ]).jacobian(vals).jacobian(idl)
  ATAinv  = (A.transpose()*A).inv()
  ATb     = A.transpose()*sp.Matrix([edges_h[0]]).jacobian(vals).transpose()
  idl_L   = (ATAinv*ATb).transpose().tolist()[0]
  L       = [edge[0].n(17) for edge in edges]
  R       = [edge[1].n(17) for edge in edges]
  coefs   = gen_coefs(NL,'a')
  p       = gen_poly(coefs)
  TVgen   = sum([ sp.integrate(sp.diff(p,x,i)**2,(x,-sp.Rational(1,2),sp.Rational(1,2))) for i in range(1,NL) ])
  return idl_L,TVlist,L,R,coeflist,TVgen


if __name__ == "__main__" :
    s2g = gen_coefs_to_gll(5,2)*gen_sten_to_coefs(5)*gen_coefs(5,'s')
    print(s2g)
    print(s2g[0,:].tolist()[0][0])
    print(s2g[1,:].tolist()[0][0])

