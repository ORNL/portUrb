
#include "YAKL.h"
#include <mpi.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

#pragma once


// Restarted GMRES(m) for use as the linear solver inside a Jacobian-Free Newton-Krylov
// (JFNK) iteration, e.g. for a matrix-free complex Poisson solve built out of stencil ops.
//
// The caller supplies `compute_Ax`, a functor with the signature
//
//     void compute_Ax( yakl::Array<real *> const & x_in , yakl::Array<real *> const & Ax_out , MPI_Comm comm )
//
// GMRES itself only ever works with flat, real-valued 1-D arrays of the local unknown
// count. Anything to do with domain decomposition -- halo exchange, splitting `x_in`
// into separate real/imaginary (or other) component arrays, gathering results back into
// `Ax_out` -- is entirely the responsibility of `compute_Ax`. This keeps GMRES agnostic to
// the structure of the underlying complex Poisson problem.
//
// All vector arithmetic that touches arrays of length x.size() is done with
// yakl::parallel_for so it runs on the GPU. Wherever an MPI-global reduction (a dot
// product or norm) is needed, the elementwise terms to be summed are first written into
// a scratch array inside a single parallel_for, and only then reduced with
// yakl::intrinsics::sum (whose result is a single small MPI_Allreduce). This avoids ever
// doing an O(len_loc) loop outside of a GPU kernel.
template <class real> requires std::is_floating_point_v<real>
struct YaklRestartedGMRES {

  // A scaling callback uses these operations to transform solution-like vectors with the right/column scaling C
  // and equation/RHS-like vectors with the left/row scaling R. The physical system A*x=b is represented internally
  // as (R*A*C)*x_scaled=R*b, with x=C*x_scaled.
  enum class ScalingOperation {
    physical_to_scaled_solution,
    scaled_to_physical_solution,
    physical_to_scaled_rhs,
    scaled_to_physical_rhs
  };

  struct Options {
    int  restart         = 30;
    int  max_iters       = 200;
    real rel_tol         = 1.e-8;
    real abs_tol         = 0;
    bool verbose         = false;
    bool reorthogonalize = true;
  };

  struct Result {
    int  iters     = 0;
    real rel_res   = 0;
    real abs_res   = 0;
    bool converged = false;
    bool converged_by_custom_test = false;
  };


  // Returns the MPI_Datatype matching the (floating point) `real` type used by the solve.
  static MPI_Datatype mpi_real_type() {
    if constexpr (std::is_same_v<real,double>) { return MPI_DOUBLE; }
    else                                       { return MPI_FLOAT;  }
  }


  // Global dot product a.b . The elementwise products are precomputed into `work`
  // (length == a.size()) inside a single parallel_for so that yakl::intrinsics::sum only
  // ever has to do a plain reduction, and the only host-side work is the MPI_Allreduce.
  static real dot( yakl::Array<real *> const & a    ,
                   yakl::Array<real *> const & b    ,
                   yakl::Array<real *> const & work ,
                   MPI_Comm                    comm ) {
    yakl::parallel_for( YAKL_AUTO_LABEL() , a.size() , KOKKOS_LAMBDA (int i) { work(i) = a(i)*b(i); });
    real loc = yakl::intrinsics::sum(work);
    real glob;
    MPI_Allreduce(&loc,&glob,1,mpi_real_type(),MPI_SUM,comm);
    return glob;
  }


  // Global 2-norm of a, built on top of dot() above.
  static real norm( yakl::Array<real *> const & a    ,
                    yakl::Array<real *> const & work ,
                    MPI_Comm                    comm ) {
    return std::sqrt( dot(a,a,work,comm) );
  }


  // Keep device lambdas outside solve(): NVCC cannot define an extended lambda in a function template when one of
  // the function's template arguments is itself a function-local lambda type, as is common for solver callbacks.
  static void form_residual( yakl::Array<real *> const & b    ,
                             yakl::Array<real *> const & r    ,
                             yakl::Array<real *> const & work ) {
    yakl::parallel_for( YAKL_AUTO_LABEL() , b.size() , KOKKOS_LAMBDA (int i) {
      real const value = b(i) - r(i);
      r(i)              = value;
      work(i)           = value*value;
    });
  }


  static void reset_to_zero_guess( yakl::Array<real *> const & x ,
                                   yakl::Array<real *> const & r ,
                                   yakl::Array<real *> const & b ) {
    yakl::parallel_for( YAKL_AUTO_LABEL() , x.size() , KOKKOS_LAMBDA (int i) {
      x(i) = 0;
      r(i) = b(i);
    });
  }


  static void scale_vector( yakl::Array<real *> const & vector , real factor ) {
    yakl::parallel_for( YAKL_AUTO_LABEL() , vector.size() , KOKKOS_LAMBDA (int i) {
      vector(i) *= factor;
    });
  }


  static void axpy( yakl::Array<real *> const & y , real alpha , yakl::Array<real *> const & x ) {
    yakl::parallel_for( YAKL_AUTO_LABEL() , y.size() , KOKKOS_LAMBDA (int i) {
      y(i) += alpha*x(i);
    });
  }


  // An optional convergence_test has signature
  //
  //   bool convergence_test(yakl::Array<real *> const & x, int iters, MPI_Comm comm)
  //
  // and is evaluated for the initial guess and each current Krylov trial solution.
  // An optional right_preconditioner has signature
  //
  //   void right_preconditioner(yakl::Array<real *> const & y,
  //                             yakl::Array<real *> const & z, MPI_Comm comm)
  //
  // and computes z = M^{-1} y. Omitting it selects the identity operation z = y.
  // An optional scaling callback has signature
  //
  //   void scaling(yakl::Array<real *> const & input,
  //                yakl::Array<real *> const & output, ScalingOperation operation)
  //
  // and applies one of the four fixed transforms described by ScalingOperation. Omitting it uses identity scaling.
  // compute_Ax, convergence_test, x, b, and a supplied right_preconditioner all remain in physical coordinates.
  template <class ComputeAx, class ConvergenceTest = std::nullptr_t, class RightPreconditioner = std::nullptr_t,
            class Scaling = std::nullptr_t>
  Result solve( yakl::Array<real *> const & x          , // initial guess (in) / solution (out)
                yakl::Array<real *> const & b          , // right hand side
                ComputeAx                   compute_Ax , // compute_Ax(x_in,Ax_out,comm) -> Ax_out = A*x_in
                Options             const & opts       ,
                MPI_Comm                    comm = MPI_COMM_WORLD,
                ConvergenceTest             convergence_test = nullptr,
                RightPreconditioner          right_preconditioner = nullptr,
                Scaling                      scaling = nullptr ) const {
    auto len_loc = x.size();
    int  rank;
    MPI_Comm_rank(comm,&rank);

    int const m = opts.restart;

    // Scratch buffer reused by every dot()/norm() call below so we never repeatedly
    // allocate/free device memory inside the Arnoldi loop.
    yakl::Array<real *> work("gmres_work",len_loc);
    yakl::Array<real *> x_trial("gmres_x_trial",len_loc);
    yakl::Array<real *> x_solver = x;
    yakl::Array<real *> b_solver = b;
    yakl::Array<real *> physical_input;
    yakl::Array<real *> physical_output;
    if constexpr (!std::is_same_v<std::decay_t<Scaling>,std::nullptr_t>) {
      x_solver       = yakl::Array<real *>("gmres_scaled_solution",len_loc);
      b_solver       = yakl::Array<real *>("gmres_scaled_rhs",len_loc);
      physical_input = yakl::Array<real *>("gmres_physical_input",len_loc);
      physical_output = yakl::Array<real *>("gmres_physical_output",len_loc);
      scaling(x,x_solver,ScalingOperation::physical_to_scaled_solution);
      scaling(b,b_solver,ScalingOperation::physical_to_scaled_rhs);
    }

    // Apply the physical matrix through a fixed two-sided scaling without exposing scaled values to compute_Ax.
    auto apply_operator = [&] (yakl::Array<real *> const & input, yakl::Array<real *> const & output,
                               MPI_Comm operator_comm) {
      if constexpr (std::is_same_v<std::decay_t<Scaling>,std::nullptr_t>) {
        compute_Ax(input,output,operator_comm);
      } else {
        scaling(input,physical_input,ScalingOperation::scaled_to_physical_solution);
        compute_Ax(physical_input,physical_output,operator_comm);
        scaling(physical_output,output,ScalingOperation::physical_to_scaled_rhs);
      }
    };

    // The preconditioner retains its physical-system contract. Its scaled-system representation is
    // C^{-1}*M^{-1}*R^{-1}; omitting it remains identity in the scaled coordinates.
    auto apply_preconditioner = [&] (yakl::Array<real *> const & input, yakl::Array<real *> const & output,
                                     MPI_Comm preconditioner_comm) {
      if constexpr (std::is_same_v<std::decay_t<RightPreconditioner>,std::nullptr_t>) {
        input.deep_copy_to(output);
      } else if constexpr (std::is_same_v<std::decay_t<Scaling>,std::nullptr_t>) {
        right_preconditioner(input,output,preconditioner_comm);
      } else {
        scaling(input,physical_input,ScalingOperation::scaled_to_physical_rhs);
        right_preconditioner(physical_input,physical_output,preconditioner_comm);
        scaling(physical_output,output,ScalingOperation::physical_to_scaled_solution);
      }
    };

    // Custom convergence tests retain their established physical-solution interface.
    auto test_convergence = [&] (yakl::Array<real *> const & input, int iters, MPI_Comm test_comm) {
      if constexpr (std::is_same_v<std::decay_t<ConvergenceTest>,std::nullptr_t>) {
        return false;
      } else if constexpr (std::is_same_v<std::decay_t<Scaling>,std::nullptr_t>) {
        return convergence_test(input,iters,test_comm);
      } else {
        scaling(input,physical_input,ScalingOperation::scaled_to_physical_solution);
        return convergence_test(physical_input,iters,test_comm);
      }
    };

    // Krylov basis vectors, stored as separate named arrays (mirrors the rest of this
    // codebase's style of one array per purpose rather than a single 2-D buffer).
    std::vector<yakl::Array<real *>> V(m+1);
    for (int k=0; k <= m; k++) V[k] = yakl::Array<real *>("gmres_V",len_loc);
    std::vector<yakl::Array<real *>> Z(m);
    for (int k=0; k < m; k++) Z[k] = yakl::Array<real *>("gmres_Z",len_loc);

    // The (m+1) x m Hessenberg matrix and Givens rotation data are all O(restart^2),
    // never touch arrays of length len_loc, and are cheap dense host-side arithmetic --
    // there is no benefit to running any of this on the GPU.
    yakl::Array<real **,Kokkos::HostSpace> H ("gmres_H" ,m+1,m);
    yakl::Array<real * ,Kokkos::HostSpace> cs("gmres_cs",m);
    yakl::Array<real * ,Kokkos::HostSpace> sn("gmres_sn",m);
    yakl::Array<real * ,Kokkos::HostSpace> g ("gmres_g" ,m+1);

    Result result;

    real const bnorm = norm(b_solver,work,comm);

    // r0 = b - A*x0 ; precompute the squared residual terms in the same kernel that
    // forms the residual so the reduction below is a plain sum.
    auto V0 = V[0];
    apply_operator(x_solver,V0,comm);
    form_residual(b_solver,V0,work);
    real loc0 = yakl::intrinsics::sum(work);
    real beta0_sq;
    MPI_Allreduce(&loc0,&beta0_sq,1,mpi_real_type(),MPI_SUM,comm);
    real beta0 = std::sqrt(beta0_sq);

    // A stale initial guess can be worse than x=0. In that case, recover the zero-guess residual without another
    // operator application. This makes rolling guesses safe even when the operator, geometry, or forcing changes.
    if (beta0 > bnorm) {
      reset_to_zero_guess(x_solver,V0,b_solver);
      beta0 = bnorm;
    }
    real const threshold = std::max(opts.abs_tol, opts.rel_tol*bnorm);

    if (opts.verbose && rank == 0) std::cout << "GMRES initial residual: " << beta0 << "\n";

    result.iters = 0;
    real beta = beta0;
    bool converged = (beta0 <= threshold);
    if constexpr (!std::is_same_v<std::decay_t<ConvergenceTest>,std::nullptr_t>) {
      result.converged_by_custom_test = test_convergence(x_solver,result.iters,comm);
      converged = converged || result.converged_by_custom_test;
    }

    while (!converged && result.iters < opts.max_iters) {
      if (result.iters > 0 && opts.verbose && rank == 0) std::cout << "*** GMRES restart, residual: " << beta << "\n";
      H  = 0;
      cs = 0;
      sn = 0;
      g  = 0;
      // Normalize the current residual (sitting in V[0]) into the first Krylov vector.
      scale_vector(V0,real(1)/beta);
      g(0) = beta;

      int j_last = -1;
      bool breakdown = false;
      bool custom_accepted = false;

      for (int j=0; j < m; j++) {
        yakl::Array<real *> Vj   = V[j];
        yakl::Array<real *> Vjp1 = V[j+1];
        yakl::Array<real *> Zj    = Z[j];
        apply_preconditioner(Vj,Zj,comm);
        apply_operator(Zj,Vjp1,comm); // Right-preconditioned Arnoldi product in solver coordinates.

        // Modified Gram-Schmidt orthogonalization against the existing basis, with an
        // optional second pass for reorthogonalization.
        int const npasses = opts.reorthogonalize ? 2 : 1;
        for (int pass=0; pass < npasses; pass++) {
          for (int i=0; i <= j; i++) {
            yakl::Array<real *> Vi = V[i];
            real h = dot(Vjp1,Vi,work,comm);
            H(i,j) += h;
            axpy(Vjp1,-h,Vi);
          }
        }

        H(j+1,j) = norm(Vjp1,work,comm);
        breakdown = H(j+1,j) <= 100*std::numeric_limits<real>::epsilon()*std::max(beta0,real(1));
        if (!breakdown) {
          real const inv = real(1)/H(j+1,j);
          scale_vector(Vjp1,inv);
        }

        // Apply the previously accumulated Givens rotations to the new column of H.
        for (int i=0; i < j; i++) {
          real const temp = cs(i)*H(i,j) + sn(i)*H(i+1,j);
          H(i+1,j)        = -sn(i)*H(i,j) + cs(i)*H(i+1,j);
          H(i,j)          = temp;
        }
        // New Givens rotation eliminating H(j+1,j).
        real const denom = std::sqrt(H(j,j)*H(j,j) + H(j+1,j)*H(j+1,j));
        cs(j) = denom > 0 ? H(j,j)  /denom : real(1);
        sn(j) = denom > 0 ? H(j+1,j)/denom : real(0);
        H(j,j)   = cs(j)*H(j,j) + sn(j)*H(j+1,j);
        H(j+1,j) = 0;
        g(j+1)   = -sn(j)*g(j);
        g(j)     =  cs(j)*g(j);

        result.iters++;
        beta = std::abs(g(j+1));
        j_last = j;

        if (opts.verbose && rank == 0) {
          std::cout << "GMRES iter " << result.iters << " residual: " << beta << "\n";
        }

        if constexpr (!std::is_same_v<std::decay_t<ConvergenceTest>,std::nullptr_t>) {
          int const k_trial = j+1;
          std::vector<real> y_trial(k_trial,real(0));
          for (int ii = k_trial-1; ii >= 0; ii--) {
            real s = g(ii);
            for (int jj = ii+1; jj < k_trial; jj++) s -= H(ii,jj)*y_trial[jj];
            y_trial[ii] = s/H(ii,ii);
          }
          x_solver.deep_copy_to(x_trial);
          for (int i = 0; i < k_trial; i++) {
            real const yi = y_trial[i];
            yakl::Array<real *> Zi = Z[i];
            axpy(x_trial,yi,Zi);
          }
          result.converged_by_custom_test = test_convergence(x_trial,result.iters,comm);
          if (result.converged_by_custom_test) {
            x_trial.deep_copy_to(x_solver);
            custom_accepted = true;
            break;
          }
        }

        if (beta <= threshold || breakdown || result.iters >= opts.max_iters) break;
      }

      // Solve the small upper-triangular least-squares system H(0:k,0:k) y = g(0:k) by
      // back substitution -- again O(k^2) host-side scalar work, independent of len_loc.
      if (!custom_accepted) {
        int const k = j_last + 1;
        std::vector<real> y(k,real(0));
        for (int ii=k-1; ii >= 0; ii--) {
          real s = g(ii);
          for (int jj=ii+1; jj < k; jj++) s -= H(ii,jj)*y[jj];
          y[ii] = s/H(ii,ii);
        }

        // Fold the right-preconditioned correction x += sum_i y[i]*Z[i] back onto the GPU solution vector.
        for (int i=0; i < k; i++) {
          real const yi = y[i];
          yakl::Array<real *> Zi = Z[i];
          axpy(x_solver,yi,Zi);
        }
      }

      // Recompute the true residual after every Krylov update. The recurrence residual can drift from b-A*x for an
      // ill-conditioned system, and an Arnoldi breakdown is convergence only when this recomputed residual is small.
      apply_operator(x_solver,V0,comm);
      form_residual(b_solver,V0,work);
      real loc = yakl::intrinsics::sum(work);
      real beta_sq;
      MPI_Allreduce(&loc,&beta_sq,1,mpi_real_type(),MPI_SUM,comm);
      beta = std::sqrt(beta_sq);
      converged = custom_accepted || (beta <= threshold);
    }

    if constexpr (!std::is_same_v<std::decay_t<Scaling>,std::nullptr_t>) {
      scaling(x_solver,x,ScalingOperation::scaled_to_physical_solution);
    }
    result.abs_res    = beta;
    result.rel_res    = bnorm > 0 ? beta/bnorm : beta;
    result.converged  = converged;
    return result;
  }

};
