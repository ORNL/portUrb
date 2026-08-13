
#include "YAKL.h"
#include <mpi.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

#pragma once


// Conjugate Gradient (CG) for use as the linear solver inside symmetric positive
// definite (SPD) systems, e.g. a matrix-free discrete Poisson solve built out of
// stencil ops.
//
// The caller supplies `apply_A`, a functor with the signature
//
//     void apply_A( yakl::Array<Scalar *> const & x_in , yakl::Array<Scalar *> const & Ax_out , MPI_Comm comm )
//
// CG itself only ever works with flat, real-valued 1-D arrays of the local unknown
// count. Anything to do with domain decomposition -- halo exchange, splitting `x_in`
// into separate real/imaginary (or other) component arrays, gathering results back into
// `Ax_out` -- is entirely the responsibility of `apply_A`. This keeps CG agnostic to the
// structure of the underlying SPD problem. `apply_A` must be symmetric positive definite
// for CG to be guaranteed to converge.
//
// All vector arithmetic that touches arrays of length x.size() is done with
// yakl::parallel_for so it runs on the GPU. Wherever an MPI-global reduction (a dot
// product or norm) is needed, the elementwise terms to be summed are first written into
// a scratch array inside a single parallel_for, and only then reduced with
// yakl::intrinsics::sum (whose result is a single small MPI_Allreduce). This avoids ever
// doing an O(len_loc) loop outside of a GPU kernel.
template <class Scalar = float> requires std::is_floating_point_v<Scalar>
struct YaklConjGrad {

  struct Options {
    int  max_iters = 200;
    Scalar rel_tol  = Scalar(1.e-8);
    Scalar abs_tol  = Scalar(0);
    bool verbose    = false;
  };

  struct Result {
    int  iters     = 0;
    Scalar rel_res = 0;
    Scalar abs_res = 0;
    bool converged = false;
  };


  // Returns the MPI_Datatype matching the floating-point scalar used by this solver instantiation.
  static MPI_Datatype mpi_real_type() {
    if constexpr (std::is_same_v<Scalar,double>) { return MPI_DOUBLE; }
    else                                         { return MPI_FLOAT;  }
  }


  // Global dot product a.b . The elementwise products are precomputed into `work`
  // (length == a.size()) inside a single parallel_for so that yakl::intrinsics::sum only
  // ever has to do a plain reduction, and the only host-side work is the MPI_Allreduce.
  static Scalar dot( yakl::Array<Scalar *> const & a    ,
                     yakl::Array<Scalar *> const & b    ,
                     yakl::Array<Scalar *> const & work ,
                     MPI_Comm                      comm ) {
    yakl::parallel_for( YAKL_AUTO_LABEL() , a.size() , KOKKOS_LAMBDA (int i) { work(i) = a(i)*b(i); });
    Scalar loc = yakl::intrinsics::sum(work);
    Scalar glob;
    MPI_Allreduce(&loc,&glob,1,mpi_real_type(),MPI_SUM,comm);
    return glob;
  }


  static Scalar norm( yakl::Array<Scalar *> const & a    ,
                      yakl::Array<Scalar *> const & work ,
                      MPI_Comm                      comm ) {
    return std::sqrt( dot(a,a,work,comm) );
  }


  template <class ApplyA>
  Result solve( yakl::Array<Scalar *> const & x       , // initial guess (in) / solution (out)
                yakl::Array<Scalar *> const & b       , // right hand side
                ApplyA                        apply_A , // apply_A(x_in,Ax_out,comm) -> Ax_out = A*x_in
                Options               const & opts    ,
                MPI_Comm                      comm = MPI_COMM_WORLD ) const {
    auto len_loc = x.size();
    int  rank;
    MPI_Comm_rank(comm,&rank);

    // Scratch buffer reused by every dot() call below so we never repeatedly
    // allocate/free device memory inside the CG loop.
    yakl::Array<Scalar *> work("cg_work",len_loc);

    // Residual, search direction, and A*p vectors -- each a separate named array
    // (mirrors the rest of this codebase's style of one array per purpose).
    yakl::Array<Scalar *> r ("cg_r" ,len_loc);
    yakl::Array<Scalar *> p ("cg_p" ,len_loc);
    yakl::Array<Scalar *> Ap("cg_Ap",len_loc);

    Result result;

    Scalar const bnorm = norm(b,work,comm);

    // r0 = b - A*x0 ; precompute the squared residual terms in the same kernel that
    // forms the residual so the reduction below is a plain sum.
    apply_A(x,Ap,comm);
    yakl::parallel_for( YAKL_AUTO_LABEL() , len_loc , KOKKOS_LAMBDA (int i) {
      Scalar ri = b(i) - Ap(i);
      r(i)     = ri;
      p(i)     = ri;
      work(i)  = ri*ri;
    });
    Scalar loc0 = yakl::intrinsics::sum(work);
    Scalar beta0_sq;
    MPI_Allreduce(&loc0,&beta0_sq,1,mpi_real_type(),MPI_SUM,comm);
    Scalar beta0 = std::sqrt(beta0_sq);

    // A stale initial guess can be worse than x=0. In that case, recover the zero-guess residual without another
    // operator application. This makes rolling guesses safe even when the operator, geometry, or forcing changes.
    if (beta0 > bnorm) {
      yakl::parallel_for( YAKL_AUTO_LABEL() , len_loc , KOKKOS_LAMBDA (int i) {
        x(i) = 0;
        r(i) = b(i);
        p(i) = b(i);
      });
      beta0_sq = bnorm*bnorm;
      beta0 = bnorm;
    }
    Scalar const threshold = std::max(opts.abs_tol,opts.rel_tol*bnorm);

    if (opts.verbose && rank == 0) std::cout << "CG initial residual: " << beta0 << "\n";

    result.iters = 0;
    Scalar beta = beta0;
    Scalar rs_old = beta0_sq;
    bool converged = (beta0 <= threshold);

    while (!converged && result.iters < opts.max_iters) {
      apply_A(p,Ap,comm); // Ap = A*p

      Scalar const pAp = dot(p,Ap,work,comm);
      // For an SPD operator, p.A.p must be positive. Its magnitude naturally approaches zero with the residual, so an
      // absolute epsilon threshold would incorrectly report breakdown near convergence.
      bool const breakdown = !std::isfinite(pAp) || pAp <= 0;
      if (breakdown) break;

      Scalar const alpha = rs_old/pAp;

      // x += alpha*p ; r -= alpha*Ap ; precompute the squared residual terms in the same
      // kernel that updates the residual so the reduction below is a plain sum.
      yakl::parallel_for( YAKL_AUTO_LABEL() , len_loc , KOKKOS_LAMBDA (int i) {
        x(i)    += alpha*p(i);
        Scalar ri = r(i) - alpha*Ap(i);
        r(i)     = ri;
        work(i)  = ri*ri;
      });
      Scalar loc = yakl::intrinsics::sum(work);
      Scalar rs_new;
      MPI_Allreduce(&loc,&rs_new,1,mpi_real_type(),MPI_SUM,comm);

      result.iters++;
      beta = std::sqrt(rs_new);

      if (opts.verbose && rank == 0) {
        std::cout << "CG iter " << result.iters << " residual: " << beta << "\n";
      }

      converged = (beta <= threshold);

      if (!converged && result.iters < opts.max_iters) {
        // p = r + (rs_new/rs_old)*p
        Scalar const gamma = rs_new/rs_old;
        yakl::parallel_for( YAKL_AUTO_LABEL() , len_loc , KOKKOS_LAMBDA (int i) {
          p(i) = r(i) + gamma*p(i);
        });
      }

      rs_old = rs_new;
    }

    // No final re-verification of the residual is performed here: by construction the
    // recurrence-based `beta` above already reflects the residual norm of the solution
    // that was just folded into `x`, so once CG reports convergence (or exhausts
    // max_iters) the residual is assumed to have been driven to (near) zero.
    result.abs_res    = beta;
    result.rel_res    = bnorm > 0 ? beta/bnorm : beta;
    result.converged  = converged;
    return result;
  }

};
