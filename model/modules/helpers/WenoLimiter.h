
#pragma once

template <class real, int ord> struct WenoLimiter;


template <class real> struct WenoLimiter<real,3> {
  static KOKKOS_INLINE_FUNCTION void value_based(SArray<real,3> v, real &L, real &R) {
    real TV0 = v(0)*v(0) - static_cast<real>(2.0000000000000000)*v(0)*v(1) + v(1)*v(1);
    real TV1 = v(1)*v(1) - static_cast<real>(2.0000000000000000)*v(1)*v(2) + v(2)*v(2);
    TV0 *= TV0;
    TV1 *= TV1;
    // Left Edge
    real w0 = static_cast<real>(0.66666666666666667)/(TV0+1.e-10);
    real w1 = static_cast<real>(0.33333333333333333)/(TV1+1.e-10);
    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1);
    w0 *= r_sm;
    w1 *= r_sm;
    real L0 = static_cast<real>(0.50000000000000000)*v(0) + static_cast<real>(0.50000000000000000)*v(1);
    real L1 = static_cast<real>(1.5000000000000000)*v(1) - static_cast<real>(0.50000000000000000)*v(2);
    L = w0*L0 + w1*L1;
    // Right Edge
    w0 = static_cast<real>(0.33333333333333333)/(TV0+1.e-10);
    w1 = static_cast<real>(0.66666666666666667)/(TV1+1.e-10);
    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1);
    w0 *= r_sm;
    w1 *= r_sm;
    real R0 = -static_cast<real>(0.50000000000000000)*v(0) + static_cast<real>(1.5000000000000000)*v(1);
    real R1 = static_cast<real>(0.50000000000000000)*v(1) + static_cast<real>(0.50000000000000000)*v(2);
    R = w0*R0 + w1*R1;
  }
  static KOKKOS_INLINE_FUNCTION void coef_based(SArray<real,3> v, real &L, real &R) {
    real c1;
    c1 = -v(0) + v(1);
    real TV0 = coefs_to_TV(c1);
    TV0 *= TV0;
    c1 = -v(1) + v(2);
    real TV1 = coefs_to_TV(c1);
    TV1 *= TV1;
    // Left Edge
    real w0 = static_cast<real>(0.66666666666666667)/(TV0+1.e-10);
    real w1 = static_cast<real>(0.33333333333333333)/(TV1+1.e-10);
    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1);
    w0 *= r_sm;
    w1 *= r_sm;
    real L0 = static_cast<real>(0.50000000000000000)*v(0) + static_cast<real>(0.50000000000000000)*v(1);
    real L1 = static_cast<real>(1.5000000000000000)*v(1) - static_cast<real>(0.50000000000000000)*v(2);
    L = w0*L0 + w1*L1;
    // Right Edge
    w0 = static_cast<real>(0.33333333333333333)/(TV0+1.e-10);
    w1 = static_cast<real>(0.66666666666666667)/(TV1+1.e-10);
    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1);
    w0 *= r_sm;
    w1 *= r_sm;
    real R0 = -static_cast<real>(0.50000000000000000)*v(0) + static_cast<real>(1.5000000000000000)*v(1);
    real R1 = static_cast<real>(0.50000000000000000)*v(1) + static_cast<real>(0.50000000000000000)*v(2);
    R = w0*R0 + w1*R1;
  }
  static KOKKOS_INLINE_FUNCTION real coefs_to_TV(real a1) {
    return a1*a1;
  }
};


template <class real> struct WenoLimiter<real,5> {
  static KOKKOS_INLINE_FUNCTION void value_based(SArray<real,5> v, real &L, real &R) {
    real TV0 = static_cast<real>(1.3333333333333333)*v(0)*v(0) - static_cast<real>(6.3333333333333333)*v(0)*v(1) + static_cast<real>(3.6666666666666667)*v(0)*v(2) + static_cast<real>(8.3333333333333333)*v(1)*v(1) - static_cast<real>(10.333333333333333)*v(1)*v(2) + static_cast<real>(3.3333333333333333)*v(2)*v(2);
    real TV1 = static_cast<real>(1.3333333333333333)*v(1)*v(1) - static_cast<real>(4.3333333333333333)*v(1)*v(2) + static_cast<real>(1.6666666666666667)*v(1)*v(3) + static_cast<real>(4.3333333333333333)*v(2)*v(2) - static_cast<real>(4.3333333333333333)*v(2)*v(3) + static_cast<real>(1.3333333333333333)*v(3)*v(3);
    real TV2 = static_cast<real>(3.3333333333333333)*v(2)*v(2) - static_cast<real>(10.333333333333333)*v(2)*v(3) + static_cast<real>(3.6666666666666667)*v(2)*v(4) + static_cast<real>(8.3333333333333333)*v(3)*v(3) - static_cast<real>(6.3333333333333333)*v(3)*v(4) + static_cast<real>(1.3333333333333333)*v(4)*v(4);
    TV0 *= TV0;
    TV1 *= TV1;
    TV2 *= TV2;
    // Left Edge
    real w0 = static_cast<real>(0.30000000000000000)/(TV0+1.e-10);
    real w1 = static_cast<real>(0.60000000000000000)/(TV1+1.e-10);
    real w2 = static_cast<real>(0.10000000000000000)/(TV2+1.e-10);
    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    real L0 = -static_cast<real>(0.16666666666666667)*v(0) + static_cast<real>(0.83333333333333333)*v(1) + static_cast<real>(0.33333333333333333)*v(2);
    real L1 = static_cast<real>(0.33333333333333333)*v(1) + static_cast<real>(0.83333333333333333)*v(2) - static_cast<real>(0.16666666666666667)*v(3);
    real L2 = static_cast<real>(1.8333333333333333)*v(2) - static_cast<real>(1.1666666666666667)*v(3) + static_cast<real>(0.33333333333333333)*v(4);
    L = w0*L0 + w1*L1 + w2*L2;
    // Right Edge
    w0 = static_cast<real>(0.10000000000000000)/(TV0+1.e-10);
    w1 = static_cast<real>(0.60000000000000000)/(TV1+1.e-10);
    w2 = static_cast<real>(0.30000000000000000)/(TV2+1.e-10);
    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    real R0 = static_cast<real>(0.33333333333333333)*v(0) - static_cast<real>(1.1666666666666667)*v(1) + static_cast<real>(1.8333333333333333)*v(2);
    real R1 = -static_cast<real>(0.16666666666666667)*v(1) + static_cast<real>(0.83333333333333333)*v(2) + static_cast<real>(0.33333333333333333)*v(3);
    real R2 = static_cast<real>(0.33333333333333333)*v(2) + static_cast<real>(0.83333333333333333)*v(3) - static_cast<real>(0.16666666666666667)*v(4);
    R = w0*R0 + w1*R1 + w2*R2;
  }
  static KOKKOS_INLINE_FUNCTION void coef_based(SArray<real,5> v, real &L, real &R) {
    real c1,c2;
    c1 = static_cast<real>(0.50000000000000000)*v(0) - static_cast<real>(2.0000000000000000)*v(1) + static_cast<real>(1.5000000000000000)*v(2);
    c2 = static_cast<real>(0.50000000000000000)*v(0) - v(1) + static_cast<real>(0.50000000000000000)*v(2);
    real TV0 = coefs_to_TV(c1,c2);
    TV0 *= TV0;
    c1 = -static_cast<real>(0.50000000000000000)*v(1) + static_cast<real>(0.50000000000000000)*v(3);
    c2 = static_cast<real>(0.50000000000000000)*v(1) - v(2) + static_cast<real>(0.50000000000000000)*v(3);
    real TV1 = coefs_to_TV(c1,c2);
    TV1 *= TV1;
    c1 = -static_cast<real>(1.5000000000000000)*v(2) + static_cast<real>(2.0000000000000000)*v(3) - static_cast<real>(0.50000000000000000)*v(4);
    c2 = static_cast<real>(0.50000000000000000)*v(2) - v(3) + static_cast<real>(0.50000000000000000)*v(4);
    real TV2 = coefs_to_TV(c1,c2);
    TV2 *= TV2;
    // Left Edge
    real w0 = static_cast<real>(0.30000000000000000)/(TV0+1.e-10);
    real w1 = static_cast<real>(0.60000000000000000)/(TV1+1.e-10);
    real w2 = static_cast<real>(0.10000000000000000)/(TV2+1.e-10);
    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    real L0 = -static_cast<real>(0.16666666666666667)*v(0) + static_cast<real>(0.83333333333333333)*v(1) + static_cast<real>(0.33333333333333333)*v(2);
    real L1 = static_cast<real>(0.33333333333333333)*v(1) + static_cast<real>(0.83333333333333333)*v(2) - static_cast<real>(0.16666666666666667)*v(3);
    real L2 = static_cast<real>(1.8333333333333333)*v(2) - static_cast<real>(1.1666666666666667)*v(3) + static_cast<real>(0.33333333333333333)*v(4);
    L = w0*L0 + w1*L1 + w2*L2;
    // Right Edge
    w0 = static_cast<real>(0.10000000000000000)/(TV0+1.e-10);
    w1 = static_cast<real>(0.60000000000000000)/(TV1+1.e-10);
    w2 = static_cast<real>(0.30000000000000000)/(TV2+1.e-10);
    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    real R0 = static_cast<real>(0.33333333333333333)*v(0) - static_cast<real>(1.1666666666666667)*v(1) + static_cast<real>(1.8333333333333333)*v(2);
    real R1 = -static_cast<real>(0.16666666666666667)*v(1) + static_cast<real>(0.83333333333333333)*v(2) + static_cast<real>(0.33333333333333333)*v(3);
    real R2 = static_cast<real>(0.33333333333333333)*v(2) + static_cast<real>(0.83333333333333333)*v(3) - static_cast<real>(0.16666666666666667)*v(4);
    R = w0*R0 + w1*R1 + w2*R2;
  }
  static KOKKOS_INLINE_FUNCTION real coefs_to_TV(real a1,real a2) {
    return a1*a1 + static_cast<real>(4.3333333333333333)*a2*a2;
  }
};


template <class real> struct WenoLimiter<real,7> {
  static KOKKOS_INLINE_FUNCTION void value_based(SArray<real,7> v, real &L, real &R) {
    real TV0 = static_cast<real>(2.2791666666666667)*v(0)*v(0) - static_cast<real>(16.175000000000000)*v(0)*v(1) + static_cast<real>(19.341666666666667)*v(0)*v(2) - static_cast<real>(7.7250000000000000)*v(0)*v(3) + static_cast<real>(29.345833333333333)*v(1)*v(1) - static_cast<real>(71.858333333333333)*v(1)*v(2) + static_cast<real>(29.341666666666667)*v(1)*v(3) + static_cast<real>(45.845833333333333)*v(2)*v(2) - static_cast<real>(39.175000000000000)*v(2)*v(3) + static_cast<real>(8.7791666666666667)*v(3)*v(3);
    real TV1 = static_cast<real>(1.1125000000000000)*v(1)*v(1) - static_cast<real>(6.8416666666666667)*v(1)*v(2) + static_cast<real>(6.6750000000000000)*v(1)*v(3) - static_cast<real>(2.0583333333333333)*v(1)*v(4) + static_cast<real>(11.845833333333333)*v(2)*v(2) - static_cast<real>(24.858333333333333)*v(2)*v(3) + static_cast<real>(8.0083333333333333)*v(2)*v(4) + static_cast<real>(14.345833333333333)*v(3)*v(3) - static_cast<real>(10.508333333333333)*v(3)*v(4) + static_cast<real>(2.2791666666666667)*v(4)*v(4);
    real TV2 = static_cast<real>(2.2791666666666667)*v(2)*v(2) - static_cast<real>(10.508333333333333)*v(2)*v(3) + static_cast<real>(8.0083333333333333)*v(2)*v(4) - static_cast<real>(2.0583333333333333)*v(2)*v(5) + static_cast<real>(14.345833333333333)*v(3)*v(3) - static_cast<real>(24.858333333333333)*v(3)*v(4) + static_cast<real>(6.6750000000000000)*v(3)*v(5) + static_cast<real>(11.845833333333333)*v(4)*v(4) - static_cast<real>(6.8416666666666667)*v(4)*v(5) + static_cast<real>(1.1125000000000000)*v(5)*v(5);
    real TV3 = static_cast<real>(8.7791666666666667)*v(3)*v(3) - static_cast<real>(39.175000000000000)*v(3)*v(4) + static_cast<real>(29.341666666666667)*v(3)*v(5) - static_cast<real>(7.7250000000000000)*v(3)*v(6) + static_cast<real>(45.845833333333333)*v(4)*v(4) - static_cast<real>(71.858333333333333)*v(4)*v(5) + static_cast<real>(19.341666666666667)*v(4)*v(6) + static_cast<real>(29.345833333333333)*v(5)*v(5) - static_cast<real>(16.175000000000000)*v(5)*v(6) + static_cast<real>(2.2791666666666667)*v(6)*v(6);
    TV0 *= TV0;
    TV1 *= TV1;
    TV2 *= TV2;
    TV3 *= TV3;
    // Left Edge
    real w0 = static_cast<real>(0.11428571428571429)/(TV0+1.e-10);
    real w1 = static_cast<real>(0.51428571428571429)/(TV1+1.e-10);
    real w2 = static_cast<real>(0.34285714285714286)/(TV2+1.e-10);
    real w3 = static_cast<real>(0.028571428571428571)/(TV3+1.e-10);
    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2 + w3);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    w3 *= r_sm;
    real L0 = static_cast<real>(0.083333333333333333)*v(0) - static_cast<real>(0.41666666666666667)*v(1) + static_cast<real>(1.0833333333333333)*v(2) + static_cast<real>(0.25000000000000000)*v(3);
    real L1 = -static_cast<real>(0.083333333333333333)*v(1) + static_cast<real>(0.58333333333333333)*v(2) + static_cast<real>(0.58333333333333333)*v(3) - static_cast<real>(0.083333333333333333)*v(4);
    real L2 = static_cast<real>(0.25000000000000000)*v(2) + static_cast<real>(1.0833333333333333)*v(3) - static_cast<real>(0.41666666666666667)*v(4) + static_cast<real>(0.083333333333333333)*v(5);
    real L3 = static_cast<real>(2.0833333333333333)*v(3) - static_cast<real>(1.9166666666666667)*v(4) + static_cast<real>(1.0833333333333333)*v(5) - static_cast<real>(0.25000000000000000)*v(6);
    L = w0*L0 + w1*L1 + w2*L2 + w3*L3;
    // Right Edge
    w0 = static_cast<real>(0.028571428571428571)/(TV0+1.e-10);
    w1 = static_cast<real>(0.34285714285714286)/(TV1+1.e-10);
    w2 = static_cast<real>(0.51428571428571429)/(TV2+1.e-10);
    w3 = static_cast<real>(0.11428571428571429)/(TV3+1.e-10);
    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2 + w3);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    w3 *= r_sm;
    real R0 = -static_cast<real>(0.25000000000000000)*v(0) + static_cast<real>(1.0833333333333333)*v(1) - static_cast<real>(1.9166666666666667)*v(2) + static_cast<real>(2.0833333333333333)*v(3);
    real R1 = static_cast<real>(0.083333333333333333)*v(1) - static_cast<real>(0.41666666666666667)*v(2) + static_cast<real>(1.0833333333333333)*v(3) + static_cast<real>(0.25000000000000000)*v(4);
    real R2 = -static_cast<real>(0.083333333333333333)*v(2) + static_cast<real>(0.58333333333333333)*v(3) + static_cast<real>(0.58333333333333333)*v(4) - static_cast<real>(0.083333333333333333)*v(5);
    real R3 = static_cast<real>(0.25000000000000000)*v(3) + static_cast<real>(1.0833333333333333)*v(4) - static_cast<real>(0.41666666666666667)*v(5) + static_cast<real>(0.083333333333333333)*v(6);
    R = w0*R0 + w1*R1 + w2*R2 + w3*R3;
  }
  static KOKKOS_INLINE_FUNCTION void coef_based(SArray<real,7> v, real &L, real &R) {
    real c1,c2,c3;
    c1 = -static_cast<real>(0.29166666666666667)*v(0) + static_cast<real>(1.3750000000000000)*v(1) - static_cast<real>(2.8750000000000000)*v(2) + static_cast<real>(1.7916666666666667)*v(3);
    c2 = -static_cast<real>(0.50000000000000000)*v(0) + static_cast<real>(2.0000000000000000)*v(1) - static_cast<real>(2.5000000000000000)*v(2) + v(3);
    c3 = -static_cast<real>(0.16666666666666667)*v(0) + static_cast<real>(0.50000000000000000)*v(1) - static_cast<real>(0.50000000000000000)*v(2) + static_cast<real>(0.16666666666666667)*v(3);
    real TV0 = coefs_to_TV(c1,c2,c3);
    TV0 *= TV0;
    c1 = static_cast<real>(0.20833333333333333)*v(1) - static_cast<real>(1.1250000000000000)*v(2) + static_cast<real>(0.62500000000000000)*v(3) + static_cast<real>(0.29166666666666667)*v(4);
    c2 = static_cast<real>(0.50000000000000000)*v(2) - v(3) + static_cast<real>(0.50000000000000000)*v(4);
    c3 = -static_cast<real>(0.16666666666666667)*v(1) + static_cast<real>(0.50000000000000000)*v(2) - static_cast<real>(0.50000000000000000)*v(3) + static_cast<real>(0.16666666666666667)*v(4);
    real TV1 = coefs_to_TV(c1,c2,c3);
    TV1 *= TV1;
    c1 = -static_cast<real>(0.29166666666666667)*v(2) - static_cast<real>(0.62500000000000000)*v(3) + static_cast<real>(1.1250000000000000)*v(4) - static_cast<real>(0.20833333333333333)*v(5);
    c2 = static_cast<real>(0.50000000000000000)*v(2) - v(3) + static_cast<real>(0.50000000000000000)*v(4);
    c3 = -static_cast<real>(0.16666666666666667)*v(2) + static_cast<real>(0.50000000000000000)*v(3) - static_cast<real>(0.50000000000000000)*v(4) + static_cast<real>(0.16666666666666667)*v(5);
    real TV2 = coefs_to_TV(c1,c2,c3);
    TV2 *= TV2;
    c1 = -static_cast<real>(1.7916666666666667)*v(3) + static_cast<real>(2.8750000000000000)*v(4) - static_cast<real>(1.3750000000000000)*v(5) + static_cast<real>(0.29166666666666667)*v(6);
    c2 = v(3) - static_cast<real>(2.5000000000000000)*v(4) + static_cast<real>(2.0000000000000000)*v(5) - static_cast<real>(0.50000000000000000)*v(6);
    c3 = -static_cast<real>(0.16666666666666667)*v(3) + static_cast<real>(0.50000000000000000)*v(4) - static_cast<real>(0.50000000000000000)*v(5) + static_cast<real>(0.16666666666666667)*v(6);
    real TV3 = coefs_to_TV(c1,c2,c3);
    TV3 *= TV3;
    // Left Edge
    real w0 = static_cast<real>(0.11428571428571429)/(TV0+1.e-10);
    real w1 = static_cast<real>(0.51428571428571429)/(TV1+1.e-10);
    real w2 = static_cast<real>(0.34285714285714286)/(TV2+1.e-10);
    real w3 = static_cast<real>(0.028571428571428571)/(TV3+1.e-10);
    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2 + w3);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    w3 *= r_sm;
    real L0 = static_cast<real>(0.083333333333333333)*v(0) - static_cast<real>(0.41666666666666667)*v(1) + static_cast<real>(1.0833333333333333)*v(2) + static_cast<real>(0.25000000000000000)*v(3);
    real L1 = -static_cast<real>(0.083333333333333333)*v(1) + static_cast<real>(0.58333333333333333)*v(2) + static_cast<real>(0.58333333333333333)*v(3) - static_cast<real>(0.083333333333333333)*v(4);
    real L2 = static_cast<real>(0.25000000000000000)*v(2) + static_cast<real>(1.0833333333333333)*v(3) - static_cast<real>(0.41666666666666667)*v(4) + static_cast<real>(0.083333333333333333)*v(5);
    real L3 = static_cast<real>(2.0833333333333333)*v(3) - static_cast<real>(1.9166666666666667)*v(4) + static_cast<real>(1.0833333333333333)*v(5) - static_cast<real>(0.25000000000000000)*v(6);
    L = w0*L0 + w1*L1 + w2*L2 + w3*L3;
    // Right Edge
    w0 = static_cast<real>(0.028571428571428571)/(TV0+1.e-10);
    w1 = static_cast<real>(0.34285714285714286)/(TV1+1.e-10);
    w2 = static_cast<real>(0.51428571428571429)/(TV2+1.e-10);
    w3 = static_cast<real>(0.11428571428571429)/(TV3+1.e-10);
    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2 + w3);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    w3 *= r_sm;
    real R0 = -static_cast<real>(0.25000000000000000)*v(0) + static_cast<real>(1.0833333333333333)*v(1) - static_cast<real>(1.9166666666666667)*v(2) + static_cast<real>(2.0833333333333333)*v(3);
    real R1 = static_cast<real>(0.083333333333333333)*v(1) - static_cast<real>(0.41666666666666667)*v(2) + static_cast<real>(1.0833333333333333)*v(3) + static_cast<real>(0.25000000000000000)*v(4);
    real R2 = -static_cast<real>(0.083333333333333333)*v(2) + static_cast<real>(0.58333333333333333)*v(3) + static_cast<real>(0.58333333333333333)*v(4) - static_cast<real>(0.083333333333333333)*v(5);
    real R3 = static_cast<real>(0.25000000000000000)*v(3) + static_cast<real>(1.0833333333333333)*v(4) - static_cast<real>(0.41666666666666667)*v(5) + static_cast<real>(0.083333333333333333)*v(6);
    R = w0*R0 + w1*R1 + w2*R2 + w3*R3;
  }
  static KOKKOS_INLINE_FUNCTION real coefs_to_TV(real a1,real a2,real a3) {
    return a1*a1 + static_cast<real>(0.50000000000000000)*a1*a3 + static_cast<real>(4.3333333333333333)*a2*a2 + static_cast<real>(39.112500000000000)*a3*a3;
  }
};


template <class real> struct WenoLimiter<real,9> {
  static KOKKOS_INLINE_FUNCTION void value_based(SArray<real,9> v, real &L, real &R) {
    real TV0 = static_cast<real>(4.4956349206349206)*v(0)*v(0) - static_cast<real>(41.369246031746032)*v(0)*v(1) + static_cast<real>(72.393452380952381)*v(0)*v(2) - static_cast<real>(57.144246031746032)*v(0)*v(3) + static_cast<real>(17.128769841269841)*v(0)*v(4) + static_cast<real>(95.825992063492063)*v(1)*v(1) - static_cast<real>(338.17380952380952)*v(1)*v(2) + static_cast<real>(269.53531746031746)*v(1)*v(3) - static_cast<real>(81.644246031746032)*v(1)*v(4) + static_cast<real>(301.86369047619048)*v(2)*v(2) - static_cast<real>(488.50714285714286)*v(2)*v(3) + static_cast<real>(150.56011904761905)*v(2)*v(4) + static_cast<real>(202.49265873015873)*v(3)*v(3) - static_cast<real>(128.86924603174603)*v(3)*v(4) + static_cast<real>(21.412301587301587)*v(4)*v(4);
    real TV1 = static_cast<real>(1.3706349206349206)*v(1)*v(1) - static_cast<real>(12.077579365079365)*v(1)*v(2) + static_cast<real>(19.685119047619048)*v(1)*v(3) - static_cast<real>(13.935912698412698)*v(1)*v(4) + static_cast<real>(3.5871031746031746)*v(1)*v(5) + static_cast<real>(27.492658730158730)*v(2)*v(2) - static_cast<real>(92.257142857142857)*v(2)*v(3) + static_cast<real>(66.868650793650794)*v(2)*v(4) - static_cast<real>(17.519246031746032)*v(2)*v(5) + static_cast<real>(80.613690476190476)*v(3)*v(3) - static_cast<real>(121.42380952380952)*v(3)*v(4) + static_cast<real>(32.768452380952381)*v(3)*v(5) + static_cast<real>(48.159325396825397)*v(4)*v(4) - static_cast<real>(27.827579365079365)*v(4)*v(5) + static_cast<real>(4.4956349206349206)*v(5)*v(5);
    real TV2 = static_cast<real>(1.3706349206349206)*v(2)*v(2) - static_cast<real>(10.119246031746032)*v(2)*v(3) + static_cast<real>(13.476785714285714)*v(2)*v(4) - static_cast<real>(7.7275793650793651)*v(2)*v(5) + static_cast<real>(1.6287698412698413)*v(2)*v(6) + static_cast<real>(20.825992063492063)*v(3)*v(3) - static_cast<real>(59.340476190476190)*v(3)*v(4) + static_cast<real>(35.535317460317460)*v(3)*v(5) - static_cast<real>(7.7275793650793651)*v(3)*v(6) + static_cast<real>(45.863690476190476)*v(4)*v(4) - static_cast<real>(59.340476190476190)*v(4)*v(5) + static_cast<real>(13.476785714285714)*v(4)*v(6) + static_cast<real>(20.825992063492063)*v(5)*v(5) - static_cast<real>(10.119246031746032)*v(5)*v(6) + static_cast<real>(1.3706349206349206)*v(6)*v(6);
    real TV3 = static_cast<real>(4.4956349206349206)*v(3)*v(3) - static_cast<real>(27.827579365079365)*v(3)*v(4) + static_cast<real>(32.768452380952381)*v(3)*v(5) - static_cast<real>(17.519246031746032)*v(3)*v(6) + static_cast<real>(3.5871031746031746)*v(3)*v(7) + static_cast<real>(48.159325396825397)*v(4)*v(4) - static_cast<real>(121.42380952380952)*v(4)*v(5) + static_cast<real>(66.868650793650794)*v(4)*v(6) - static_cast<real>(13.935912698412698)*v(4)*v(7) + static_cast<real>(80.613690476190476)*v(5)*v(5) - static_cast<real>(92.257142857142857)*v(5)*v(6) + static_cast<real>(19.685119047619048)*v(5)*v(7) + static_cast<real>(27.492658730158730)*v(6)*v(6) - static_cast<real>(12.077579365079365)*v(6)*v(7) + static_cast<real>(1.3706349206349206)*v(7)*v(7);
    real TV4 = static_cast<real>(21.412301587301587)*v(4)*v(4) - static_cast<real>(128.86924603174603)*v(4)*v(5) + static_cast<real>(150.56011904761905)*v(4)*v(6) - static_cast<real>(81.644246031746032)*v(4)*v(7) + static_cast<real>(17.128769841269841)*v(4)*v(8) + static_cast<real>(202.49265873015873)*v(5)*v(5) - static_cast<real>(488.50714285714286)*v(5)*v(6) + static_cast<real>(269.53531746031746)*v(5)*v(7) - static_cast<real>(57.144246031746032)*v(5)*v(8) + static_cast<real>(301.86369047619048)*v(6)*v(6) - static_cast<real>(338.17380952380952)*v(6)*v(7) + static_cast<real>(72.393452380952381)*v(6)*v(8) + static_cast<real>(95.825992063492063)*v(7)*v(7) - static_cast<real>(41.369246031746032)*v(7)*v(8) + static_cast<real>(4.4956349206349206)*v(8)*v(8);
    TV0 *= TV0;
    TV1 *= TV1;
    TV2 *= TV2;
    TV3 *= TV3;
    TV4 *= TV4;
    // Left Edge
    real w0 = static_cast<real>(0.039682539682539683)/(TV0+1.e-10);
    real w1 = static_cast<real>(0.31746031746031746)/(TV1+1.e-10);
    real w2 = static_cast<real>(0.47619047619047619)/(TV2+1.e-10);
    real w3 = static_cast<real>(0.15873015873015873)/(TV3+1.e-10);
    real w4 = static_cast<real>(0.0079365079365079365)/(TV4+1.e-10);
    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2 + w3 + w4);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    w3 *= r_sm;
    w4 *= r_sm;
    real L0 = -static_cast<real>(0.050000000000000000)*v(0) + static_cast<real>(0.28333333333333333)*v(1) - static_cast<real>(0.71666666666666667)*v(2) + static_cast<real>(1.2833333333333333)*v(3) + static_cast<real>(0.20000000000000000)*v(4);
    real L1 = static_cast<real>(0.033333333333333333)*v(1) - static_cast<real>(0.21666666666666667)*v(2) + static_cast<real>(0.78333333333333333)*v(3) + static_cast<real>(0.45000000000000000)*v(4) - static_cast<real>(0.050000000000000000)*v(5);
    real L2 = -static_cast<real>(0.050000000000000000)*v(2) + static_cast<real>(0.45000000000000000)*v(3) + static_cast<real>(0.78333333333333333)*v(4) - static_cast<real>(0.21666666666666667)*v(5) + static_cast<real>(0.033333333333333333)*v(6);
    real L3 = static_cast<real>(0.20000000000000000)*v(3) + static_cast<real>(1.2833333333333333)*v(4) - static_cast<real>(0.71666666666666667)*v(5) + static_cast<real>(0.28333333333333333)*v(6) - static_cast<real>(0.050000000000000000)*v(7);
    real L4 = static_cast<real>(2.2833333333333333)*v(4) - static_cast<real>(2.7166666666666667)*v(5) + static_cast<real>(2.2833333333333333)*v(6) - static_cast<real>(1.0500000000000000)*v(7) + static_cast<real>(0.20000000000000000)*v(8);
    L = w0*L0 + w1*L1 + w2*L2 + w3*L3 + w4*L4;
    // Right Edge
    w0 = static_cast<real>(0.0079365079365079365)/(TV0+1.e-10);
    w1 = static_cast<real>(0.15873015873015873)/(TV1+1.e-10);
    w2 = static_cast<real>(0.47619047619047619)/(TV2+1.e-10);
    w3 = static_cast<real>(0.31746031746031746)/(TV3+1.e-10);
    w4 = static_cast<real>(0.039682539682539683)/(TV4+1.e-10);
    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2 + w3 + w4);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    w3 *= r_sm;
    w4 *= r_sm;
    real R0 = static_cast<real>(0.20000000000000000)*v(0) - static_cast<real>(1.0500000000000000)*v(1) + static_cast<real>(2.2833333333333333)*v(2) - static_cast<real>(2.7166666666666667)*v(3) + static_cast<real>(2.2833333333333333)*v(4);
    real R1 = -static_cast<real>(0.050000000000000000)*v(1) + static_cast<real>(0.28333333333333333)*v(2) - static_cast<real>(0.71666666666666667)*v(3) + static_cast<real>(1.2833333333333333)*v(4) + static_cast<real>(0.20000000000000000)*v(5);
    real R2 = static_cast<real>(0.033333333333333333)*v(2) - static_cast<real>(0.21666666666666667)*v(3) + static_cast<real>(0.78333333333333333)*v(4) + static_cast<real>(0.45000000000000000)*v(5) - static_cast<real>(0.050000000000000000)*v(6);
    real R3 = -static_cast<real>(0.050000000000000000)*v(3) + static_cast<real>(0.45000000000000000)*v(4) + static_cast<real>(0.78333333333333333)*v(5) - static_cast<real>(0.21666666666666667)*v(6) + static_cast<real>(0.033333333333333333)*v(7);
    real R4 = static_cast<real>(0.20000000000000000)*v(4) + static_cast<real>(1.2833333333333333)*v(5) - static_cast<real>(0.71666666666666667)*v(6) + static_cast<real>(0.28333333333333333)*v(7) - static_cast<real>(0.050000000000000000)*v(8);
    R = w0*R0 + w1*R1 + w2*R2 + w3*R3 + w4*R4;
  }
  static KOKKOS_INLINE_FUNCTION void coef_based(SArray<real,9> v, real &L, real &R) {
    real c1,c2,c3,c4;
    c1 = static_cast<real>(0.18750000000000000)*v(0) - static_cast<real>(1.0416666666666667)*v(1) + static_cast<real>(2.5000000000000000)*v(2) - static_cast<real>(3.6250000000000000)*v(3) + static_cast<real>(1.9791666666666667)*v(4);
    c2 = static_cast<real>(0.43750000000000000)*v(0) - static_cast<real>(2.2500000000000000)*v(1) + static_cast<real>(4.6250000000000000)*v(2) - static_cast<real>(4.2500000000000000)*v(3) + static_cast<real>(1.4375000000000000)*v(4);
    c3 = static_cast<real>(0.25000000000000000)*v(0) - static_cast<real>(1.1666666666666667)*v(1) + static_cast<real>(2.0000000000000000)*v(2) - static_cast<real>(1.5000000000000000)*v(3) + static_cast<real>(0.41666666666666667)*v(4);
    c4 = static_cast<real>(0.041666666666666667)*v(0) - static_cast<real>(0.16666666666666667)*v(1) + static_cast<real>(0.25000000000000000)*v(2) - static_cast<real>(0.16666666666666667)*v(3) + static_cast<real>(0.041666666666666667)*v(4);
    real TV0 = coefs_to_TV(c1,c2,c3,c4);
    TV0 *= TV0;
    c1 = -static_cast<real>(0.10416666666666667)*v(1) + static_cast<real>(0.62500000000000000)*v(2) - static_cast<real>(1.7500000000000000)*v(3) + static_cast<real>(1.0416666666666667)*v(4) + static_cast<real>(0.18750000000000000)*v(5);
    c2 = -static_cast<real>(0.062500000000000000)*v(1) + static_cast<real>(0.25000000000000000)*v(2) + static_cast<real>(0.12500000000000000)*v(3) - static_cast<real>(0.75000000000000000)*v(4) + static_cast<real>(0.43750000000000000)*v(5);
    c3 = static_cast<real>(0.083333333333333333)*v(1) - static_cast<real>(0.50000000000000000)*v(2) + v(3) - static_cast<real>(0.83333333333333333)*v(4) + static_cast<real>(0.25000000000000000)*v(5);
    c4 = static_cast<real>(0.041666666666666667)*v(1) - static_cast<real>(0.16666666666666667)*v(2) + static_cast<real>(0.25000000000000000)*v(3) - static_cast<real>(0.16666666666666667)*v(4) + static_cast<real>(0.041666666666666667)*v(5);
    real TV1 = coefs_to_TV(c1,c2,c3,c4);
    TV1 *= TV1;
    c1 = static_cast<real>(0.10416666666666667)*v(2) - static_cast<real>(0.70833333333333333)*v(3) + static_cast<real>(0.70833333333333333)*v(5) - static_cast<real>(0.10416666666666667)*v(6);
    c2 = -static_cast<real>(0.062500000000000000)*v(2) + static_cast<real>(0.75000000000000000)*v(3) - static_cast<real>(1.3750000000000000)*v(4) + static_cast<real>(0.75000000000000000)*v(5) - static_cast<real>(0.062500000000000000)*v(6);
    c3 = -static_cast<real>(0.083333333333333333)*v(2) + static_cast<real>(0.16666666666666667)*v(3) - static_cast<real>(0.16666666666666667)*v(5) + static_cast<real>(0.083333333333333333)*v(6);
    c4 = static_cast<real>(0.041666666666666667)*v(2) - static_cast<real>(0.16666666666666667)*v(3) + static_cast<real>(0.25000000000000000)*v(4) - static_cast<real>(0.16666666666666667)*v(5) + static_cast<real>(0.041666666666666667)*v(6);
    real TV2 = coefs_to_TV(c1,c2,c3,c4);
    TV2 *= TV2;
    c1 = -static_cast<real>(0.18750000000000000)*v(3) - static_cast<real>(1.0416666666666667)*v(4) + static_cast<real>(1.7500000000000000)*v(5) - static_cast<real>(0.62500000000000000)*v(6) + static_cast<real>(0.10416666666666667)*v(7);
    c2 = static_cast<real>(0.43750000000000000)*v(3) - static_cast<real>(0.75000000000000000)*v(4) + static_cast<real>(0.12500000000000000)*v(5) + static_cast<real>(0.25000000000000000)*v(6) - static_cast<real>(0.062500000000000000)*v(7);
    c3 = -static_cast<real>(0.25000000000000000)*v(3) + static_cast<real>(0.83333333333333333)*v(4) - v(5) + static_cast<real>(0.50000000000000000)*v(6) - static_cast<real>(0.083333333333333333)*v(7);
    c4 = static_cast<real>(0.041666666666666667)*v(3) - static_cast<real>(0.16666666666666667)*v(4) + static_cast<real>(0.25000000000000000)*v(5) - static_cast<real>(0.16666666666666667)*v(6) + static_cast<real>(0.041666666666666667)*v(7);
    real TV3 = coefs_to_TV(c1,c2,c3,c4);
    TV3 *= TV3;
    c1 = -static_cast<real>(1.9791666666666667)*v(4) + static_cast<real>(3.6250000000000000)*v(5) - static_cast<real>(2.5000000000000000)*v(6) + static_cast<real>(1.0416666666666667)*v(7) - static_cast<real>(0.18750000000000000)*v(8);
    c2 = static_cast<real>(1.4375000000000000)*v(4) - static_cast<real>(4.2500000000000000)*v(5) + static_cast<real>(4.6250000000000000)*v(6) - static_cast<real>(2.2500000000000000)*v(7) + static_cast<real>(0.43750000000000000)*v(8);
    c3 = -static_cast<real>(0.41666666666666667)*v(4) + static_cast<real>(1.5000000000000000)*v(5) - static_cast<real>(2.0000000000000000)*v(6) + static_cast<real>(1.1666666666666667)*v(7) - static_cast<real>(0.25000000000000000)*v(8);
    c4 = static_cast<real>(0.041666666666666667)*v(4) - static_cast<real>(0.16666666666666667)*v(5) + static_cast<real>(0.25000000000000000)*v(6) - static_cast<real>(0.16666666666666667)*v(7) + static_cast<real>(0.041666666666666667)*v(8);
    real TV4 = coefs_to_TV(c1,c2,c3,c4);
    TV4 *= TV4;
    // Left Edge
    real w0 = static_cast<real>(0.039682539682539683)/(TV0+1.e-10);
    real w1 = static_cast<real>(0.31746031746031746)/(TV1+1.e-10);
    real w2 = static_cast<real>(0.47619047619047619)/(TV2+1.e-10);
    real w3 = static_cast<real>(0.15873015873015873)/(TV3+1.e-10);
    real w4 = static_cast<real>(0.0079365079365079365)/(TV4+1.e-10);
    real r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2 + w3 + w4);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    w3 *= r_sm;
    w4 *= r_sm;
    real L0 = -static_cast<real>(0.050000000000000000)*v(0) + static_cast<real>(0.28333333333333333)*v(1) - static_cast<real>(0.71666666666666667)*v(2) + static_cast<real>(1.2833333333333333)*v(3) + static_cast<real>(0.20000000000000000)*v(4);
    real L1 = static_cast<real>(0.033333333333333333)*v(1) - static_cast<real>(0.21666666666666667)*v(2) + static_cast<real>(0.78333333333333333)*v(3) + static_cast<real>(0.45000000000000000)*v(4) - static_cast<real>(0.050000000000000000)*v(5);
    real L2 = -static_cast<real>(0.050000000000000000)*v(2) + static_cast<real>(0.45000000000000000)*v(3) + static_cast<real>(0.78333333333333333)*v(4) - static_cast<real>(0.21666666666666667)*v(5) + static_cast<real>(0.033333333333333333)*v(6);
    real L3 = static_cast<real>(0.20000000000000000)*v(3) + static_cast<real>(1.2833333333333333)*v(4) - static_cast<real>(0.71666666666666667)*v(5) + static_cast<real>(0.28333333333333333)*v(6) - static_cast<real>(0.050000000000000000)*v(7);
    real L4 = static_cast<real>(2.2833333333333333)*v(4) - static_cast<real>(2.7166666666666667)*v(5) + static_cast<real>(2.2833333333333333)*v(6) - static_cast<real>(1.0500000000000000)*v(7) + static_cast<real>(0.20000000000000000)*v(8);
    L = w0*L0 + w1*L1 + w2*L2 + w3*L3 + w4*L4;
    // Right Edge
    w0 = static_cast<real>(0.0079365079365079365)/(TV0+1.e-10);
    w1 = static_cast<real>(0.15873015873015873)/(TV1+1.e-10);
    w2 = static_cast<real>(0.47619047619047619)/(TV2+1.e-10);
    w3 = static_cast<real>(0.31746031746031746)/(TV3+1.e-10);
    w4 = static_cast<real>(0.039682539682539683)/(TV4+1.e-10);
    r_sm = static_cast<real>(1.)/std::max(static_cast<real>(1.e-10),w0 + w1 + w2 + w3 + w4);
    w0 *= r_sm;
    w1 *= r_sm;
    w2 *= r_sm;
    w3 *= r_sm;
    w4 *= r_sm;
    real R0 = static_cast<real>(0.20000000000000000)*v(0) - static_cast<real>(1.0500000000000000)*v(1) + static_cast<real>(2.2833333333333333)*v(2) - static_cast<real>(2.7166666666666667)*v(3) + static_cast<real>(2.2833333333333333)*v(4);
    real R1 = -static_cast<real>(0.050000000000000000)*v(1) + static_cast<real>(0.28333333333333333)*v(2) - static_cast<real>(0.71666666666666667)*v(3) + static_cast<real>(1.2833333333333333)*v(4) + static_cast<real>(0.20000000000000000)*v(5);
    real R2 = static_cast<real>(0.033333333333333333)*v(2) - static_cast<real>(0.21666666666666667)*v(3) + static_cast<real>(0.78333333333333333)*v(4) + static_cast<real>(0.45000000000000000)*v(5) - static_cast<real>(0.050000000000000000)*v(6);
    real R3 = -static_cast<real>(0.050000000000000000)*v(3) + static_cast<real>(0.45000000000000000)*v(4) + static_cast<real>(0.78333333333333333)*v(5) - static_cast<real>(0.21666666666666667)*v(6) + static_cast<real>(0.033333333333333333)*v(7);
    real R4 = static_cast<real>(0.20000000000000000)*v(4) + static_cast<real>(1.2833333333333333)*v(5) - static_cast<real>(0.71666666666666667)*v(6) + static_cast<real>(0.28333333333333333)*v(7) - static_cast<real>(0.050000000000000000)*v(8);
    R = w0*R0 + w1*R1 + w2*R2 + w3*R3 + w4*R4;
  }
  static KOKKOS_INLINE_FUNCTION real coefs_to_TV(real a1,real a2,real a3,real a4) {
    return a1*a1 + static_cast<real>(0.50000000000000000)*a1*a3 + static_cast<real>(4.3333333333333333)*a2*a2 + static_cast<real>(4.2000000000000000)*a2*a4 + static_cast<real>(39.112500000000000)*a3*a3 + static_cast<real>(625.83571428571429)*a4*a4;
  }
};
