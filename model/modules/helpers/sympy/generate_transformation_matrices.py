
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

print("}\n")

