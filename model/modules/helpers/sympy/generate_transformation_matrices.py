
import transformations as tr

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
print("  ///////////////")
print("  // WENO")
print("  ///////////////")
print(f"  template <class F> KOKKOS_INLINE_FUNCTION void weno( yakl::SArray<F,1,1,2> & s2g   ,")
print(f"                                                       yakl::SArray<F,1>     & idl_L ,")
print(f"                                                       yakl::SArray<F,1,1,1> & ATV   ,")
print(f"                                                       yakl::SArray<F,1,1,1> & ATV_L ,")
print(f"                                                       yakl::SArray<F,1,1,1> & ATV_R ) "+"{")
for i in range(2) :
  print(f"    s2g  (0,0,{i}) = static_cast<F>(1);")
print(f"    idl_L(0)       = static_cast<F>(1);")
print(f"    ATV  (0,0,0)   = static_cast<F>(0);")
print(f"    ATV_L(0,0,0)   = static_cast<F>(0);")
print(f"    ATV_R(0,0,0)   = static_cast<F>(0);")
print("  }")
for N in range(3,Nmax+1) :
  if (N%2==1) :
    NL = (N+1)//2
    s2g,idl_L,ATV,ATV_L,ATV_R = tr.gen_weno_sten_to_edges_idl_TV(N)
    print(f"  template <class F> KOKKOS_INLINE_FUNCTION void weno( yakl::SArray<F,{NL},2,{NL}> & s2g ,")
    print(f"                                                       yakl::SArray<F,{NL}> & idl_L ,")
    print(f"                                                       yakl::SArray<F,{NL},{NL},{NL}> & ATV ) "+"{")
    for k in range(NL) :
      for j in range(2) :
        for i in range(NL) :
          print(f"    s2g({k},{j},{i}) = static_cast<F>({s2g[k][j][i].n(prec)});")
    for k in range(NL) :
      print(f"    idl_L({k}) = static_cast<F>({idl_L[k].n(prec)});")
    for k in range(NL) :
      for j in range(NL) :
        for i in range(NL) :
          print(f"    ATV({k},{j},{i}) = static_cast<F>({ATV[k][j][i].n(prec)});")
    print("  }")

print("}\n")

