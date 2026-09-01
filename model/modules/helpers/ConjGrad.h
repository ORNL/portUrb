
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
// An optional right_preconditioner has signature
//
//     void right_preconditioner(yakl::Array<Scalar *> const & r,
//                               yakl::Array<Scalar *> const & z, MPI_Comm comm)
//
// and computes z=M^{-1}r for a fixed symmetric positive-definite M. Although the callback follows the same
// input/output convention as a right preconditioner, CG uses the symmetry-preserving preconditioned recurrence based
// on r.z; it does not form the generally nonsymmetric right-preconditioned operator A*M^{-1}.
//
// All vector arithmetic and local reductions execute on the GPU. Global dot products
// reduce directly into a scalar without materializing an elementwise-product array.
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

  // Caller-owned storage lets repeated solves reuse the same device allocations. Array
  // handles are shallow, so DataManager-backed views can be supplied directly.
  struct Workspace {
    yakl::Array<Scalar *> r;
    yakl::Array<Scalar *> z;
    yakl::Array<Scalar *> p;
    yakl::Array<Scalar *> Ap;
    yakl::Array<Scalar *> s;
  };


  // Returns the MPI_Datatype matching the floating-point scalar used by this solver instantiation.
  static MPI_Datatype mpi_real_type() {
    if constexpr (std::is_same_v<Scalar,double>) { return MPI_DOUBLE; }
    else                                         { return MPI_FLOAT;  }
  }


  // Global dot product a.b. Kokkos performs the complete local reduction without a
  // full-size product workspace or an extra device-memory pass.
  static Scalar dot( yakl::Array<Scalar *> const & a ,
                     yakl::Array<Scalar *> const & b ,
                     MPI_Comm                      comm ) {
    Scalar loc = 0;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_local_dot_reduce");
    Kokkos::parallel_reduce(YAKL_AUTO_LABEL(),Kokkos::RangePolicy<>(0,a.size()),
                            KOKKOS_LAMBDA (int i, Scalar &sum) { sum += a(i)*b(i); },loc);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_local_dot_reduce");
    Scalar glob;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_mpi_allreduce_dot");
    MPI_Allreduce(&loc,&glob,1,mpi_real_type(),MPI_SUM,comm);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_mpi_allreduce_dot");
    return glob;
  }


  static Scalar norm( yakl::Array<Scalar *> const & a , MPI_Comm comm ) {
    return std::sqrt(dot(a,a,comm));
  }


  static void local_norm_dot( yakl::Array<Scalar *> const & r ,
                              yakl::Array<Scalar *> const & z ,
                              Scalar                         & rr,
                              Scalar                         & rz ) {
    rr = 0;
    rz = 0;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_local_norm_dot_reduce");
    Kokkos::parallel_reduce(YAKL_AUTO_LABEL(),Kokkos::RangePolicy<>(0,r.size()),
                            KOKKOS_LAMBDA (int i, Scalar &rr_sum, Scalar &rz_sum) {
      rr_sum += r(i)*r(i);
      rz_sum += r(i)*z(i);
    },Kokkos::Sum<Scalar>(rr),Kokkos::Sum<Scalar>(rz));
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_local_norm_dot_reduce");
  }


  static Scalar local_dot( yakl::Array<Scalar *> const & a ,
                           yakl::Array<Scalar *> const & b ) {
    Scalar result = 0;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_local_dot_reduce");
    Kokkos::parallel_reduce(YAKL_AUTO_LABEL(),Kokkos::RangePolicy<>(0,a.size()),
                            KOKKOS_LAMBDA (int i, Scalar &sum) { sum += a(i)*b(i); },result);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_local_dot_reduce");
    return result;
  }


  static Scalar form_residual( yakl::Array<Scalar *> const & b  ,
                               yakl::Array<Scalar *> const & Ax ,
                               yakl::Array<Scalar *> const & r  ) {
    Scalar result = 0;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_form_residual_reduce");
    Kokkos::parallel_reduce(YAKL_AUTO_LABEL(),Kokkos::RangePolicy<>(0,b.size()),
                            KOKKOS_LAMBDA (int i, Scalar &sum) {
      Scalar const ri = b(i)-Ax(i);
      r(i) = ri;
      sum += ri*ri;
    },result);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_form_residual_reduce");
    return result;
  }


  static void use_zero_guess( yakl::Array<Scalar *> const & x ,
                              yakl::Array<Scalar *> const & b ,
                              yakl::Array<Scalar *> const & r ) {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_vector_update");
    yakl::parallel_for(YAKL_AUTO_LABEL(),x.size(),KOKKOS_LAMBDA (int i) {
      x(i) = 0;
      r(i) = b(i);
    });
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_vector_update");
  }


  static void initialize_recurrence_vectors( yakl::Array<Scalar *> const & p  ,
                                             yakl::Array<Scalar *> const & s  ,
                                             yakl::Array<Scalar *> const & u  ,
                                             yakl::Array<Scalar *> const & Au ) {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_vector_update");
    yakl::parallel_for(YAKL_AUTO_LABEL(),p.size(),KOKKOS_LAMBDA (int i) {
      p(i) = u (i);
      s(i) = Au(i);
    });
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_vector_update");
  }


  static void update_solution_and_residual( yakl::Array<Scalar *> const & x ,
                                            yakl::Array<Scalar *> const & r ,
                                            yakl::Array<Scalar *> const & p ,
                                            yakl::Array<Scalar *> const & s ,
                                            Scalar                         alpha ) {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_vector_update");
    yakl::parallel_for(YAKL_AUTO_LABEL(),x.size(),KOKKOS_LAMBDA (int i) {
      x(i) += alpha*p(i);
      r(i) -= alpha*s(i);
    });
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_vector_update");
  }


  static void update_recurrence_vectors( yakl::Array<Scalar *> const & p    ,
                                         yakl::Array<Scalar *> const & s    ,
                                         yakl::Array<Scalar *> const & u    ,
                                         yakl::Array<Scalar *> const & Au   ,
                                         Scalar                         beta ) {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_vector_update");
    yakl::parallel_for(YAKL_AUTO_LABEL(),p.size(),KOKKOS_LAMBDA (int i) {
      p(i) = u (i) + beta*p(i);
      s(i) = Au(i) + beta*s(i);
    });
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_vector_update");
  }


  template <class ApplyA, class RightPreconditioner = std::nullptr_t, class ApplyAAndDot = std::nullptr_t>
  Result solve( yakl::Array<Scalar *> const & x       , // initial guess (in) / solution (out)
                yakl::Array<Scalar *> const & b       , // right hand side
                ApplyA                        apply_A , // apply_A(x_in,Ax_out,comm) -> Ax_out = A*x_in
                Workspace              const & workspace,
                Options               const & opts    ,
                MPI_Comm                      comm = MPI_COMM_WORLD,
                RightPreconditioner           right_preconditioner = nullptr,
                ApplyAAndDot                  apply_A_and_dot = nullptr ) const {
    auto len_loc = x.size();
    int  rank;
    MPI_Comm_rank(comm,&rank);

    auto r  = workspace.r;
    auto z  = workspace.z;
    auto p  = workspace.p;
    auto Ap = workspace.Ap;
    auto s  = workspace.s;
    if (r.size() != len_loc || z.size() != len_loc || p.size() != len_loc ||
        Ap.size() != len_loc || s.size() != len_loc) {
      Kokkos::abort("ERROR: CG workspace size does not match the local vector size");
    }

    Result result;

    Scalar const bnorm = norm(b,comm);

    // Form r0 and its norm in one GPU reduction kernel.
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_apply_operator");
    apply_A(x,Ap,comm);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_apply_operator");
    Scalar const loc0 = form_residual(b,Ap,r);
    Scalar beta0_sq;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_mpi_allreduce_residual");
    MPI_Allreduce(&loc0,&beta0_sq,1,mpi_real_type(),MPI_SUM,comm);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_mpi_allreduce_residual");
    Scalar beta0 = std::sqrt(beta0_sq);

    // A stale initial guess can be worse than x=0. In that case, recover the zero-guess residual without another
    // operator application. This makes rolling guesses safe even when the operator, geometry, or forcing changes.
    if (beta0 > bnorm) {
      use_zero_guess(x,b,r);
      beta0_sq = bnorm*bnorm;
      beta0 = bnorm;
    }
    Scalar const threshold = std::max(opts.abs_tol,opts.rel_tol*bnorm);

    if (opts.verbose && rank == 0) std::cout << "CG initial residual: " << beta0 << "\n";

    result.iters = 0;
    Scalar beta = beta0;
    bool converged = (beta0 <= threshold);

    auto apply_preconditioner = [&] (yakl::Array<Scalar *> const & input,
                                     yakl::Array<Scalar *> const & output) {
      if constexpr (std::is_same_v<std::decay_t<RightPreconditioner>,std::nullptr_t>) {
        return input;
      } else {
        if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_apply_preconditioner");
        right_preconditioner(input,output,comm);
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_apply_preconditioner");
        return output;
      }
    };

    auto apply_A_with_local_dot = [&] (yakl::Array<Scalar *> const & input,
                                       yakl::Array<Scalar *> const & output) {
      if constexpr (std::is_same_v<std::decay_t<ApplyAAndDot>,std::nullptr_t>) {
        if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_apply_operator");
        apply_A(input,output,comm);
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_apply_operator");
        return local_dot(input,output);
      } else {
        if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_apply_operator_and_dot");
        Scalar const result = apply_A_and_dot(input,output,comm);
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_apply_operator_and_dot");
        return result;
      }
    };

    Scalar gamma = 0;
    Scalar alpha = 0;
    auto restart_recurrence = [&] () {
      auto u = apply_preconditioner(r,z);
      Scalar const delta_loc = apply_A_with_local_dot(u,Ap);
      Scalar rr_loc;
      Scalar gamma_loc;
      local_norm_dot(r,u,rr_loc,gamma_loc);
      Scalar loc[3] = {rr_loc,gamma_loc,delta_loc};
      Scalar glob[3];
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_mpi_allreduce_recurrence");
      MPI_Allreduce(loc,glob,3,mpi_real_type(),MPI_SUM,comm);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_mpi_allreduce_recurrence");
      beta = std::sqrt(glob[0]);
      gamma = glob[1];
      Scalar const delta = glob[2];
      if (!std::isfinite(gamma) || gamma <= 0 || !std::isfinite(delta) || delta <= 0) return false;
      alpha = gamma/delta;
      if (!std::isfinite(alpha) || alpha <= 0) return false;
      initialize_recurrence_vectors(p,s,u,Ap);
      return true;
    };

    auto replace_with_true_residual = [&] () {
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_apply_operator");
      apply_A(x,Ap,comm);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_apply_operator");
      Scalar const true_loc = form_residual(b,Ap,r);
      Scalar true_sq;
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_mpi_allreduce_residual");
      MPI_Allreduce(&true_loc,&true_sq,1,mpi_real_type(),MPI_SUM,comm);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_mpi_allreduce_residual");
      beta = std::sqrt(true_sq);
      if (opts.verbose && rank == 0) std::cout << "CG true residual: " << beta << "\n";
      return beta <= threshold;
    };

    if (!converged && !restart_recurrence()) {
      result.abs_res = beta;
      result.rel_res = bnorm > 0 ? beta/bnorm : beta;
      return result;
    }

    while (!converged && result.iters < opts.max_iters) {
      // Chronopoulos-Gear PCG delays convergence testing until after M^-1 and A are applied. This permits r.r, r.z,
      // and z.Az to share one global reduction while retaining one operator and one preconditioner application per step.
      update_solution_and_residual(x,r,p,s,alpha);

      result.iters++;
      auto u = apply_preconditioner(r,z);
      Scalar const delta_loc = apply_A_with_local_dot(u,Ap);
      Scalar rr_loc;
      Scalar gamma_loc;
      local_norm_dot(r,u,rr_loc,gamma_loc);
      Scalar loc[3] = {rr_loc,gamma_loc,delta_loc};
      Scalar glob[3];
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("cg_mpi_allreduce_recurrence");
      MPI_Allreduce(loc,glob,3,mpi_real_type(),MPI_SUM,comm);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("cg_mpi_allreduce_recurrence");
      beta = std::sqrt(glob[0]);
      Scalar const gamma_new = glob[1];
      Scalar const delta_new = glob[2];

      if (opts.verbose && rank == 0) {
        std::cout << "CG iter " << result.iters << " residual: " << beta << "\n";
      }

      bool residual_replaced = false;
      if (beta <= threshold) {
        // The recursively updated residual loses its exact relation to b-A*x in finite precision. Verify convergence
        // with a fresh operator application; if it has drifted above tolerance, restart PCG from that true residual.
        converged = replace_with_true_residual();
        residual_replaced = !converged;
      }

      if (!converged && result.iters < opts.max_iters) {
        if (residual_replaced) {
          if (!restart_recurrence()) break;
        } else {
          bool breakdown = !std::isfinite(gamma_new) || gamma_new <= 0 || !std::isfinite(delta_new);
          Scalar recurrence_beta = 0;
          Scalar denominator = 0;
          if (!breakdown) {
            recurrence_beta = gamma_new/gamma;
            denominator = delta_new-recurrence_beta*gamma_new/alpha;
            breakdown = !std::isfinite(recurrence_beta) || !std::isfinite(denominator) || denominator <= 0;
          }
          if (breakdown) {
            converged = replace_with_true_residual();
            if (!converged && !restart_recurrence()) break;
          } else {
            alpha = gamma_new/denominator;
            if (!std::isfinite(alpha) || alpha <= 0) {
              converged = replace_with_true_residual();
              if (!converged && !restart_recurrence()) break;
            } else {
              update_recurrence_vectors(p,s,u,Ap,recurrence_beta);
              gamma = gamma_new;
            }
          }
        }
      }
    }

    // Report and accept convergence only from the true residual, including recurrence breakdown and iteration exhaustion.
    if (!converged) converged = replace_with_true_residual();
    result.abs_res    = beta;
    result.rel_res    = bnorm > 0 ? beta/bnorm : beta;
    result.converged  = converged;
    return result;
  }

};
