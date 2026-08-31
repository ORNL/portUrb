#pragma once

#include "main_header.h"
#include "coupler.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>


namespace modules {


// Fixed-work aggregation multigrid for the second-order immersed pressure graph used as an approximation to the
// high-order projection operator. Aggregates are grown only through active fluid-fluid faces. Every coarse matrix is
// formed by the Galerkin product P^T*A*P with piecewise-constant prolongation, so coarse levels never reinterpret or
// threshold the immersed geometry. Local aggregation stops at MPI boundaries; Galerkin edges retain the inter-rank
// connectivity. A small globally replicated coarse solve supplies the global correction missing from one-level Schwarz.
template <class Scalar = float> requires std::is_floating_point_v<Scalar>
class ConnectivityGalerkinMultigrid {
public:
  struct HostLevel {
    int nlocal = 0;
    int nglobal = 0;
    int offset = 0;
    std::vector<int> offsets;
    std::vector<std::unordered_map<int,Scalar>> rows;
    std::vector<Scalar> mass;
  };

  struct CommPlan {
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::vector<int> recv_gids;
    yakl::Array<int *> send_indices;
    yakl::Array<Scalar *> send_values;
    yakl::Array<Scalar *> recv_values;
    yakl::Array<Scalar *,Kokkos::HostSpace> send_values_host;
    yakl::Array<Scalar *,Kokkos::HostSpace> recv_values_host;
  };

  struct Level {
    int nlocal = 0;
    int nglobal = 0;
    int offset = 0;
    yakl::Array<int *> row_offsets;
    yakl::Array<int *> columns;
    yakl::Array<Scalar *> values;
    yakl::Array<Scalar *> diagonal;
    yakl::Array<Scalar *> mass;
    yakl::Array<int *> parent;
    yakl::Array<Scalar *> x;
    yakl::Array<Scalar *> b;
    yakl::Array<Scalar *> residual;
    yakl::Array<Scalar *> Ax;
    yakl::Array<Scalar *,Kokkos::HostSpace> b_host;
    yakl::Array<Scalar *,Kokkos::HostSpace> x_host;
    CommPlan comm;
  };

private:
  std::vector<Level> levels_;
  yakl::Array<int *> fine_cell_to_node_;
  std::vector<int> coarse_counts_;
  std::vector<int> coarse_displs_;
  std::vector<Scalar> coarse_matrix_;
  std::vector<Scalar> coarse_mass_;
  std::vector<int> coarse_component_;
  mutable std::vector<Scalar> coarse_factor_;
  mutable std::vector<Scalar> coarse_rhs_;
  mutable Scalar factored_shift_ = std::numeric_limits<Scalar>::quiet_NaN();
  bool direct_coarse_solve_ = false;
  MPI_Comm comm_ = MPI_COMM_NULL;
  int rank_ = 0;
  int nranks_ = 1;
  int fine_size_ = 0;
  int vcycles_ = 1;
  int pre_smooth_ = 1;
  int post_smooth_ = 1;
  int coarse_smooth_ = 16;
  Scalar jacobi_weight_ = Scalar(2)/Scalar(3);
  bool initialized_ = false;


  static MPI_Datatype mpi_scalar_type() {
    if constexpr (std::is_same_v<Scalar,double>) return MPI_DOUBLE;
    return MPI_FLOAT;
  }


  static std::vector<int> displacements(std::vector<int> const &counts) {
    std::vector<int> displs(counts.size(),0);
    for (int i = 1; i < counts.size(); i++) displs[i] = displs[i-1]+counts[i-1];
    return displs;
  }


  static int owner_of(int gid, std::vector<int> const &offsets) {
    auto const iter = std::upper_bound(offsets.begin(),offsets.end(),gid);
    return std::max(0,static_cast<int>(iter-offsets.begin())-1);
  }


  HostLevel make_host_level(std::vector<std::unordered_map<int,Scalar>> rows,
                            std::vector<Scalar> mass) const {
    HostLevel level;
    level.nlocal = static_cast<int>(rows.size());
    std::vector<int> counts(nranks_);
    MPI_Allgather(&level.nlocal,1,MPI_INT,counts.data(),1,MPI_INT,comm_);
    level.offsets.resize(nranks_+1,0);
    for (int rank = 0; rank < nranks_; rank++) level.offsets[rank+1] = level.offsets[rank]+counts[rank];
    level.offset = level.offsets[rank_];
    level.nglobal = level.offsets.back();
    level.rows = std::move(rows);
    level.mass = std::move(mass);
    return level;
  }


  CommPlan make_comm_plan(HostLevel const &host, std::vector<int> &encoded_columns,
                          std::vector<int> const &global_columns) const {
    CommPlan plan;
    plan.send_counts.assign(nranks_,0);
    plan.recv_counts.assign(nranks_,0);
    std::vector<std::vector<int>> requested_by_rank(nranks_);
    for (int gid : global_columns) {
      int const owner = owner_of(gid,host.offsets);
      if (owner != rank_) requested_by_rank[owner].push_back(gid);
    }
    for (int rank = 0; rank < nranks_; rank++) {
      auto &requested = requested_by_rank[rank];
      std::sort(requested.begin(),requested.end());
      requested.erase(std::unique(requested.begin(),requested.end()),requested.end());
      plan.recv_counts[rank] = static_cast<int>(requested.size());
    }
    plan.recv_displs = displacements(plan.recv_counts);
    int const recv_total = std::accumulate(plan.recv_counts.begin(),plan.recv_counts.end(),0);
    std::vector<int> requested_gids(recv_total);
    for (int rank = 0; rank < nranks_; rank++) {
      std::copy(requested_by_rank[rank].begin(),requested_by_rank[rank].end(),
                requested_gids.begin()+plan.recv_displs[rank]);
    }
    plan.recv_gids = requested_gids;
    MPI_Alltoall(plan.recv_counts.data(),1,MPI_INT,plan.send_counts.data(),1,MPI_INT,comm_);
    plan.send_displs = displacements(plan.send_counts);
    int const send_total = std::accumulate(plan.send_counts.begin(),plan.send_counts.end(),0);
    std::vector<int> send_gids(send_total);
    MPI_Alltoallv(requested_gids.data(),plan.recv_counts.data(),plan.recv_displs.data(),MPI_INT,
                  send_gids.data(),plan.send_counts.data(),plan.send_displs.data(),MPI_INT,comm_);

    std::unordered_map<int,int> imported_index;
    for (int index = 0; index < requested_gids.size(); index++) imported_index[requested_gids[index]] = index;
    encoded_columns.resize(global_columns.size());
    for (int entry = 0; entry < global_columns.size(); entry++) {
      int const gid = global_columns[entry];
      int const owner = owner_of(gid,host.offsets);
      encoded_columns[entry] = owner == rank_ ? gid-host.offset : -1-imported_index.at(gid);
    }

    intHost1d send_indices_host("multigrid_send_indices_host",send_total);
    for (int index = 0; index < send_total; index++) send_indices_host(index) = send_gids[index]-host.offset;
    plan.send_indices = send_indices_host.createDeviceCopy();
    plan.send_values = yakl::Array<Scalar *>("multigrid_send_values",send_total);
    plan.recv_values = yakl::Array<Scalar *>("multigrid_recv_values",recv_total);
    plan.send_values_host = yakl::Array<Scalar *,Kokkos::HostSpace>("multigrid_send_values_host",send_total);
    plan.recv_values_host = yakl::Array<Scalar *,Kokkos::HostSpace>("multigrid_recv_values_host",recv_total);
    return plan;
  }


  Level make_device_level(HostLevel const &host) const {
    Level level;
    level.nlocal = host.nlocal;
    level.nglobal = host.nglobal;
    level.offset = host.offset;
    std::vector<int> row_offsets(host.nlocal+1,0);
    std::vector<int> global_columns;
    std::vector<Scalar> values;
    std::vector<Scalar> diagonal(host.nlocal,0);
    for (int row = 0; row < host.nlocal; row++) {
      std::vector<std::pair<int,Scalar>> entries(host.rows[row].begin(),host.rows[row].end());
      std::sort(entries.begin(),entries.end(),[] (auto const &left, auto const &right) {
        return left.first < right.first;
      });
      for (auto const &[column,value] : entries) {
        if (value == 0) continue;
        global_columns.push_back(column);
        values.push_back(value);
        if (column == host.offset+row) diagonal[row] += value;
      }
      row_offsets[row+1] = static_cast<int>(values.size());
    }
    std::vector<int> encoded_columns;
    level.comm = make_comm_plan(host,encoded_columns,global_columns);

    intHost1d row_offsets_host("multigrid_row_offsets_host",row_offsets.size());
    intHost1d columns_host("multigrid_columns_host",encoded_columns.size());
    yakl::Array<Scalar *,Kokkos::HostSpace> values_host("multigrid_values_host",values.size());
    yakl::Array<Scalar *,Kokkos::HostSpace> diagonal_host("multigrid_diagonal_host",diagonal.size());
    yakl::Array<Scalar *,Kokkos::HostSpace> mass_host("multigrid_mass_host",host.mass.size());
    for (int index = 0; index < row_offsets.size(); index++) row_offsets_host(index) = row_offsets[index];
    for (int index = 0; index < values.size(); index++) {
      columns_host(index) = encoded_columns[index];
      values_host(index) = values[index];
    }
    for (int index = 0; index < host.nlocal; index++) {
      diagonal_host(index) = diagonal[index];
      mass_host(index) = host.mass[index];
    }
    level.row_offsets = row_offsets_host.createDeviceCopy();
    level.columns = columns_host.createDeviceCopy();
    level.values = values_host.createDeviceCopy();
    level.diagonal = diagonal_host.createDeviceCopy();
    level.mass = mass_host.createDeviceCopy();
    level.x = yakl::Array<Scalar *>("multigrid_x",host.nlocal);
    level.b = yakl::Array<Scalar *>("multigrid_b",host.nlocal);
    level.residual = yakl::Array<Scalar *>("multigrid_residual",host.nlocal);
    level.Ax = yakl::Array<Scalar *>("multigrid_Ax",host.nlocal);
    level.b_host = yakl::Array<Scalar *,Kokkos::HostSpace>("multigrid_b_host",host.nlocal);
    level.x_host = yakl::Array<Scalar *,Kokkos::HostSpace>("multigrid_x_host",host.nlocal);
    return level;
  }


  std::vector<int> exchange_parent_gids(HostLevel const &host, CommPlan const &plan,
                                        std::vector<int> const &parent_global) const {
    int const send_total = std::accumulate(plan.send_counts.begin(),plan.send_counts.end(),0);
    int const recv_total = std::accumulate(plan.recv_counts.begin(),plan.recv_counts.end(),0);
    auto send_indices_host = plan.send_indices.createHostCopy();
    std::vector<int> send_values(send_total);
    std::vector<int> recv_values(recv_total);
    for (int index = 0; index < send_total; index++) send_values[index] = parent_global[send_indices_host(index)];
    MPI_Alltoallv(send_values.data(),plan.send_counts.data(),plan.send_displs.data(),MPI_INT,
                  recv_values.data(),plan.recv_counts.data(),plan.recv_displs.data(),MPI_INT,comm_);
    return recv_values;
  }


  std::vector<int> aggregate(HostLevel const &host, int target_size, int &num_aggregates) const {
    std::vector<int> parent(host.nlocal,-1);
    num_aggregates = 0;
    for (int seed = 0; seed < host.nlocal; seed++) {
      if (parent[seed] >= 0) continue;
      std::queue<int> frontier;
      parent[seed] = num_aggregates;
      frontier.push(seed);
      int aggregate_size = 1;
      while (!frontier.empty() && aggregate_size < target_size) {
        int const row = frontier.front();
        frontier.pop();
        std::vector<int> neighbors;
        for (auto const &[gid,value] : host.rows[row]) {
          if (value >= 0 || gid < host.offset || gid >= host.offset+host.nlocal) continue;
          int const neighbor = gid-host.offset;
          if (parent[neighbor] < 0) neighbors.push_back(neighbor);
        }
        std::sort(neighbors.begin(),neighbors.end());
        for (int neighbor : neighbors) {
          if (aggregate_size >= target_size) break;
          parent[neighbor] = num_aggregates;
          frontier.push(neighbor);
          aggregate_size++;
        }
      }
      num_aggregates++;
    }
    return parent;
  }


  HostLevel galerkin_coarsen(HostLevel const &fine, CommPlan const &fine_comm,
                             std::vector<int> const &parent, int num_aggregates) const {
    std::vector<int> aggregate_counts(nranks_);
    MPI_Allgather(&num_aggregates,1,MPI_INT,aggregate_counts.data(),1,MPI_INT,comm_);
    std::vector<int> aggregate_offsets(nranks_+1,0);
    for (int rank = 0; rank < nranks_; rank++) {
      aggregate_offsets[rank+1] = aggregate_offsets[rank]+aggregate_counts[rank];
    }
    std::vector<int> parent_global(fine.nlocal);
    for (int node = 0; node < fine.nlocal; node++) parent_global[node] = aggregate_offsets[rank_]+parent[node];
    std::vector<int> remote_parent = exchange_parent_gids(fine,fine_comm,parent_global);
    std::vector<std::unordered_map<int,Scalar>> coarse_rows(num_aggregates);
    std::vector<Scalar> coarse_mass(num_aggregates,0);
    for (int row = 0; row < fine.nlocal; row++) {
      int const coarse_row = parent[row];
      coarse_mass[coarse_row] += fine.mass[row];
      for (auto const &[column_gid,value] : fine.rows[row]) {
        int coarse_column_gid;
        if (column_gid >= fine.offset && column_gid < fine.offset+fine.nlocal) {
          coarse_column_gid = parent_global[column_gid-fine.offset];
        } else {
          auto const iter = std::lower_bound(fine_comm.recv_gids.begin(),fine_comm.recv_gids.end(),column_gid);
          coarse_column_gid = iter != fine_comm.recv_gids.end() && *iter == column_gid ?
                              remote_parent[iter-fine_comm.recv_gids.begin()] : -1;
          if (coarse_column_gid < 0) endrun("ERROR: multigrid Galerkin setup could not map a remote column");
        }
        coarse_rows[coarse_row][coarse_column_gid] += value;
      }
    }
    return make_host_level(std::move(coarse_rows),std::move(coarse_mass));
  }


public:
  // nvcc requires member functions enclosing extended device lambdas to be publicly nameable.
  void exchange(Level &level, yakl::Array<Scalar *> const &x) const {
    int const send_total = level.comm.send_values.size();
    if (send_total > 0) {
      auto send_indices = level.comm.send_indices;
      auto send_values = level.comm.send_values;
      yakl::parallel_for(YAKL_AUTO_LABEL(),send_total,KOKKOS_LAMBDA (int index) {
        send_values(index) = x(send_indices(index));
      });
    }
    level.comm.send_values.deep_copy_to(level.comm.send_values_host);
    MPI_Alltoallv(level.comm.send_values_host.data(),level.comm.send_counts.data(),level.comm.send_displs.data(),
                  mpi_scalar_type(),level.comm.recv_values_host.data(),level.comm.recv_counts.data(),
                  level.comm.recv_displs.data(),mpi_scalar_type(),comm_);
    level.comm.recv_values_host.deep_copy_to(level.comm.recv_values);
  }


  void matvec(Level &level, yakl::Array<Scalar *> const &x, yakl::Array<Scalar *> const &Ax,
              Scalar shift) const {
    exchange(level,x);
    auto row_offsets = level.row_offsets;
    auto columns = level.columns;
    auto values = level.values;
    auto mass = level.mass;
    auto imported = level.comm.recv_values;
    yakl::parallel_for(YAKL_AUTO_LABEL(),level.nlocal,KOKKOS_LAMBDA (int row) {
      Scalar value = shift*mass(row)*x(row);
      for (int entry = row_offsets(row); entry < row_offsets(row+1); entry++) {
        int const column = columns(entry);
        Scalar const x_column = column >= 0 ? x(column) : imported(-1-column);
        value += values(entry)*x_column;
      }
      Ax(row) = value;
    });
  }


  void smooth(Level &level, int iterations, Scalar shift) const {
    auto x = level.x;
    auto b = level.b;
    auto Ax = level.Ax;
    auto diagonal = level.diagonal;
    auto mass = level.mass;
    Scalar const weight = jacobi_weight_;
    for (int iteration = 0; iteration < iterations; iteration++) {
      matvec(level,x,Ax,shift);
      yakl::parallel_for(YAKL_AUTO_LABEL(),level.nlocal,KOKKOS_LAMBDA (int row) {
        Scalar const diag = diagonal(row)+shift*mass(row);
        if (diag > std::numeric_limits<Scalar>::min()) x(row) += weight*(b(row)-Ax(row))/diag;
      });
    }
  }


  void factor_coarse_matrix(Scalar shift) const {
    if (!direct_coarse_solve_) endrun("ERROR: requested a direct solve for an iterative multigrid coarse level");
    if (shift == factored_shift_) return;
    int const n = static_cast<int>(coarse_mass_.size());
    size_t const n_size = static_cast<size_t>(n);
    coarse_factor_ = coarse_matrix_;
    for (int row = 0; row < n; row++) {
      size_t const diagonal_index = static_cast<size_t>(row)*n_size+row;
      coarse_factor_[diagonal_index] += shift*coarse_mass_[row];
    }
    if (shift == 0) {
      std::vector<int> component_sizes;
      for (int component : coarse_component_) {
        if (component >= component_sizes.size()) component_sizes.resize(component+1,0);
        component_sizes[component]++;
      }
      for (int row = 0; row < n; row++) {
        int const component = coarse_component_[row];
        Scalar const correction = Scalar(1)/static_cast<Scalar>(component_sizes[component]);
        for (int column = 0; column < n; column++) {
          if (coarse_component_[column] == component) {
            coarse_factor_[static_cast<size_t>(row)*n_size+column] += correction;
          }
        }
      }
    }
    for (int column = 0; column < n; column++) {
      size_t const column_offset = static_cast<size_t>(column)*n_size;
      Scalar diagonal = coarse_factor_[column_offset+column];
      for (int inner = 0; inner < column; inner++) {
        Scalar const value = coarse_factor_[column_offset+inner];
        diagonal -= value*value;
      }
      if (!(diagonal > Scalar(100)*std::numeric_limits<Scalar>::epsilon())) {
        endrun("ERROR: multigrid coarse Galerkin matrix is not positive definite");
      }
      coarse_factor_[column_offset+column] = std::sqrt(diagonal);
      for (int row = column+1; row < n; row++) {
        size_t const row_offset = static_cast<size_t>(row)*n_size;
        Scalar value = coarse_factor_[row_offset+column];
        for (int inner = 0; inner < column; inner++) {
          value -= coarse_factor_[row_offset+inner]*coarse_factor_[column_offset+inner];
        }
        coarse_factor_[row_offset+column] = value/coarse_factor_[column_offset+column];
      }
    }
    factored_shift_ = shift;
  }


  void coarse_solve(Level &level, Scalar shift) const {
    if (!direct_coarse_solve_) {
      // A fixed number of zero-initialized weighted-Jacobi steps is a symmetric polynomial coarse inverse. It keeps CG
      // valid while avoiding a replicated O(n^2) matrix when decomposition boundaries or disconnected geometry prevent
      // the hierarchy from reaching the requested direct-solve size.
      level.x = 0;
      smooth(level,coarse_smooth_,shift);
      return;
    }
    factor_coarse_matrix(shift);
    level.b.deep_copy_to(level.b_host);
    int const n = static_cast<int>(coarse_mass_.size());
    auto &rhs = coarse_rhs_;
    size_t const n_size = static_cast<size_t>(n);
    MPI_Allgatherv(level.b_host.data(),level.nlocal,mpi_scalar_type(),rhs.data(),coarse_counts_.data(),
                   coarse_displs_.data(),mpi_scalar_type(),comm_);
    for (int row = 0; row < n; row++) {
      size_t const row_offset = static_cast<size_t>(row)*n_size;
      for (int column = 0; column < row; column++) rhs[row] -= coarse_factor_[row_offset+column]*rhs[column];
      rhs[row] /= coarse_factor_[row_offset+row];
    }
    for (int row = n-1; row >= 0; row--) {
      for (int column = row+1; column < n; column++) {
        rhs[row] -= coarse_factor_[static_cast<size_t>(column)*n_size+row]*rhs[column];
      }
      rhs[row] /= coarse_factor_[static_cast<size_t>(row)*n_size+row];
    }
    for (int row = 0; row < level.nlocal; row++) level.x_host(row) = rhs[level.offset+row];
    level.x_host.deep_copy_to(level.x);
  }


  void vcycle(int level_index, Scalar shift) const {
    Level &level = const_cast<Level &>(levels_[level_index]);
    if (level_index+1 == levels_.size()) {
      coarse_solve(level,shift);
      return;
    }
    smooth(level,pre_smooth_,shift);
    matvec(level,level.x,level.Ax,shift);
    auto residual = level.residual;
    auto b = level.b;
    auto Ax = level.Ax;
    yakl::parallel_for(YAKL_AUTO_LABEL(),level.nlocal,KOKKOS_LAMBDA (int row) {
      residual(row) = b(row)-Ax(row);
    });
    Level &coarse = const_cast<Level &>(levels_[level_index+1]);
    coarse.b = 0;
    coarse.x = 0;
    auto parent = level.parent;
    auto coarse_b = coarse.b;
    yakl::parallel_for(YAKL_AUTO_LABEL(),level.nlocal,KOKKOS_LAMBDA (int row) {
      Kokkos::atomic_add(&coarse_b(parent(row)),residual(row));
    });
    vcycle(level_index+1,shift);
    auto x = level.x;
    auto coarse_x = coarse.x;
    yakl::parallel_for(YAKL_AUTO_LABEL(),level.nlocal,KOKKOS_LAMBDA (int row) {
      x(row) += coarse_x(parent(row));
    });
    smooth(level,post_smooth_,shift);
  }


  struct Options {
    int vcycles = 1;
    int pre_smooth = 1;
    int post_smooth = 1;
    int aggregate_size = 8;
    int max_levels = 24;
    int coarse_max_dofs = 256;
    int coarse_smooth = 16;
    Scalar jacobi_weight = Scalar(2)/Scalar(3);
  };


  bool initialized() const { return initialized_; }


  void initialize(core::Coupler &coupler, Options const &options) {
    if (initialized_) endrun("ERROR: ConnectivityGalerkinMultigrid initialized more than once");
    if (options.vcycles <= 0 || options.pre_smooth < 0 || options.post_smooth < 0 ||
        options.pre_smooth != options.post_smooth ||
        options.aggregate_size < 2 || options.max_levels < 2 || options.coarse_max_dofs <= 0 ||
        options.coarse_smooth <= 0 ||
        !(options.jacobi_weight > 0 && options.jacobi_weight < 1)) {
      endrun("ERROR: invalid connectivity Galerkin multigrid options; CG requires equal pre/post smoothing counts");
    }
    comm_ = coupler.get_parallel_comm().get_mpi_comm();
    MPI_Comm_rank(comm_,&rank_);
    MPI_Comm_size(comm_,&nranks_);
    vcycles_ = options.vcycles;
    pre_smooth_ = options.pre_smooth;
    post_smooth_ = options.post_smooth;
    coarse_smooth_ = options.coarse_smooth;
    jacobi_weight_ = options.jacobi_weight;

    int const nx = coupler.get_nx();
    int const ny = coupler.get_ny();
    int const nz = coupler.get_nz();
    int const nx_global = coupler.get_nx_glob();
    int const ny_global = coupler.get_ny_glob();
    int const i_begin = coupler.get_i_beg();
    int const j_begin = coupler.get_j_beg();
    fine_size_ = nx*ny*nz;
    auto const &dm = coupler.get_data_manager_readonly();
    auto mask_host = dm.get<int const,3>("dycore_anelastic_fluid_mask").createHostCopy();
    auto dz_host = coupler.get_dz().createHostCopy();
    auto metric_host = dm.get<real const,1>("dycore_metjac_edges").createHostCopy();

    std::vector<int> cell_to_node(fine_size_,-1);
    int num_fluid = 0;
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          int const cell = i+nx*(j+ny*k);
          if (mask_host(k,j,i) == 1) cell_to_node[cell] = num_fluid++;
        }
      }
    }
    int fluid_offset = 0;
    MPI_Exscan(&num_fluid,&fluid_offset,1,MPI_INT,MPI_SUM,comm_);
    if (rank_ == 0) fluid_offset = 0;
    int global_fluid = 0;
    MPI_Allreduce(&num_fluid,&global_fluid,1,MPI_INT,MPI_SUM,comm_);
    if (global_fluid <= 0) endrun("ERROR: multigrid hierarchy has no fluid cells");

    intHost1d cell_to_node_host("multigrid_cell_to_node_host",fine_size_);
    for (int cell = 0; cell < fine_size_; cell++) cell_to_node_host(cell) = cell_to_node[cell];
    fine_cell_to_node_ = cell_to_node_host.createDeviceCopy();
    auto fine_cell_to_node = fine_cell_to_node_;

    yakl::Array<int ****> gid_halos("multigrid_fine_gid_halos",1,nz+2,ny+2,nx+2);
    gid_halos = -1;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,nx),KOKKOS_LAMBDA (int k, int j, int i) {
      int const cell = i+nx*(j+ny*k);
      int const node = fine_cell_to_node(cell);
      gid_halos(0,1+k,1+j,1+i) = node >= 0 ? fluid_offset+node : -1;
    });
    coupler.halo_exchange(gid_halos,1);
    auto gids_host = gid_halos.createHostCopy();
    bool const periodic_x = coupler.get_option<std::string>("bc_x1") == "periodic";
    bool const periodic_y = coupler.get_option<std::string>("bc_y1") == "periodic";
    bool const periodic_z = coupler.get_option<std::string>("bc_z1") == "periodic";
    Scalar const x_weight = Scalar(1)/(static_cast<Scalar>(coupler.get_dx())*coupler.get_dx());
    Scalar const y_weight = Scalar(1)/(static_cast<Scalar>(coupler.get_dy())*coupler.get_dy());
    std::vector<std::unordered_map<int,Scalar>> rows(num_fluid);
    std::vector<Scalar> mass(num_fluid,Scalar(1));
    auto add_edge = [&] (int row, int neighbor_gid, Scalar weight) {
      if (neighbor_gid < 0) return;
      rows[row][fluid_offset+row] += weight;
      rows[row][neighbor_gid] -= weight;
    };
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          int const cell = i+nx*(j+ny*k);
          int const row = cell_to_node[cell];
          if (row < 0) continue;
          int const i_global = i_begin+i;
          int const j_global = j_begin+j;
          if (periodic_x || i_global > 0) add_edge(row,gids_host(0,1+k,1+j,i),x_weight);
          if (periodic_x || i_global+1 < nx_global) add_edge(row,gids_host(0,1+k,1+j,i+2),x_weight);
          if (periodic_y || j_global > 0) add_edge(row,gids_host(0,1+k,j,1+i),y_weight);
          if (periodic_y || j_global+1 < ny_global) add_edge(row,gids_host(0,1+k,j+2,1+i),y_weight);
          int const k_minus = k > 0 ? k-1 : nz-1;
          int const k_plus = k+1 < nz ? k+1 : 0;
          if (k > 0 || periodic_z) {
            int const neighbor = cell_to_node[i+nx*(j+ny*k_minus)];
            Scalar const weight = Scalar(1)/(static_cast<Scalar>(dz_host(k))*metric_host(k));
            add_edge(row,neighbor >= 0 ? fluid_offset+neighbor : -1,weight);
          }
          if (k+1 < nz || periodic_z) {
            int const neighbor = cell_to_node[i+nx*(j+ny*k_plus)];
            Scalar const weight = Scalar(1)/(static_cast<Scalar>(dz_host(k))*metric_host(k+1));
            add_edge(row,neighbor >= 0 ? fluid_offset+neighbor : -1,weight);
          }
        }
      }
    }

    HostLevel host = make_host_level(std::move(rows),std::move(mass));
    for (int level_index = 0; level_index < options.max_levels; level_index++) {
      Level device = make_device_level(host);
      bool const make_coarse = host.nglobal > options.coarse_max_dofs && level_index+1 < options.max_levels;
      if (!make_coarse) {
        levels_.push_back(std::move(device));
        break;
      }
      int num_aggregates = 0;
      std::vector<int> parent = aggregate(host,options.aggregate_size,num_aggregates);
      int global_aggregates = 0;
      MPI_Allreduce(&num_aggregates,&global_aggregates,1,MPI_INT,MPI_SUM,comm_);
      if (global_aggregates >= host.nglobal) {
        levels_.push_back(std::move(device));
        break;
      }
      intHost1d parent_host("multigrid_parent_host",parent.size());
      for (int node = 0; node < parent.size(); node++) parent_host(node) = parent[node];
      device.parent = parent_host.createDeviceCopy();
      HostLevel coarse = galerkin_coarsen(host,device.comm,parent,num_aggregates);
      levels_.push_back(std::move(device));
      host = std::move(coarse);
    }
    if (levels_.empty()) endrun("ERROR: multigrid hierarchy construction produced no levels");

    Level const &coarse = levels_.back();
    coarse_counts_.resize(nranks_);
    MPI_Allgather(&coarse.nlocal,1,MPI_INT,coarse_counts_.data(),1,MPI_INT,comm_);
    coarse_displs_ = displacements(coarse_counts_);
    int const coarse_n = coarse.nglobal;
    HostLevel const &coarse_host = host;
    // coarse_max_dofs is a hard memory-safety limit for the replicated dense matrix, not merely a coarsening target.
    direct_coarse_solve_ = coarse_n <= options.coarse_max_dofs;
    if (direct_coarse_solve_) {
      size_t const coarse_n_size = static_cast<size_t>(coarse_n);
      size_t const local_matrix_size = static_cast<size_t>(coarse.nlocal)*coarse_n_size;
      std::vector<Scalar> local_dense(local_matrix_size,0);
      for (int row = 0; row < coarse.nlocal; row++) {
        size_t const row_offset = static_cast<size_t>(row)*coarse_n_size;
        for (auto const &[column,value] : coarse_host.rows[row]) local_dense[row_offset+column] += value;
      }
      std::vector<int> matrix_counts(nranks_);
      std::vector<int> matrix_displs(nranks_);
      for (int rank = 0; rank < nranks_; rank++) matrix_counts[rank] = coarse_counts_[rank]*coarse_n;
      matrix_displs = displacements(matrix_counts);
      coarse_matrix_.resize(coarse_n_size*coarse_n_size);
      MPI_Allgatherv(local_dense.data(),static_cast<int>(local_dense.size()),mpi_scalar_type(),coarse_matrix_.data(),
                     matrix_counts.data(),matrix_displs.data(),mpi_scalar_type(),comm_);
      coarse_mass_.resize(coarse_n);
      coarse_rhs_.resize(coarse_n);
      MPI_Allgatherv(coarse_host.mass.data(),coarse.nlocal,mpi_scalar_type(),coarse_mass_.data(),
                     coarse_counts_.data(),coarse_displs_.data(),mpi_scalar_type(),comm_);

      coarse_component_.assign(coarse_n,-1);
      int component = 0;
      for (int seed = 0; seed < coarse_n; seed++) {
        if (coarse_component_[seed] >= 0) continue;
        std::queue<int> frontier;
        coarse_component_[seed] = component;
        frontier.push(seed);
        while (!frontier.empty()) {
          int const row = frontier.front();
          frontier.pop();
          size_t const row_offset = static_cast<size_t>(row)*coarse_n_size;
          for (int column = 0; column < coarse_n; column++) {
            if (row == column || coarse_matrix_[row_offset+column] >= 0 || coarse_component_[column] >= 0) continue;
            coarse_component_[column] = component;
            frontier.push(column);
          }
        }
        component++;
      }
    }
    coupler.set_option<int>("dycore_anelastic_multigrid_levels",levels_.size());
    coupler.set_option<int>("dycore_anelastic_multigrid_coarse_dofs",coarse_n);
    coupler.set_option<bool>("dycore_anelastic_multigrid_direct_coarse_solve",direct_coarse_solve_);
    initialized_ = true;
  }


  void apply(yakl::Array<Scalar *> const &r, yakl::Array<Scalar *> const &z,
             Scalar screening_inverse_length_squared, Scalar dt) const {
    if (!initialized_) endrun("ERROR: applying an uninitialized connectivity Galerkin multigrid preconditioner");
    if (r.size() != fine_size_ || z.size() != fine_size_) {
      endrun("ERROR: multigrid input/output size does not match the initialized fine grid");
    }
    Level &fine = const_cast<Level &>(levels_.front());
    auto cell_to_node = fine_cell_to_node_;
    auto fine_b = fine.b;
    Scalar const inverse_dt = Scalar(1)/dt;
    fine.x = 0;
    yakl::parallel_for(YAKL_AUTO_LABEL(),fine_size_,KOKKOS_LAMBDA (int cell) {
      int const node = cell_to_node(cell);
      if (node >= 0) fine_b(node) = r(cell)*inverse_dt;
      z(cell) = 0;
    });
    for (int cycle = 0; cycle < vcycles_; cycle++) vcycle(0,screening_inverse_length_squared);
    auto fine_x = fine.x;
    yakl::parallel_for(YAKL_AUTO_LABEL(),fine_size_,KOKKOS_LAMBDA (int cell) {
      int const node = cell_to_node(cell);
      if (node >= 0) z(cell) = fine_x(node);
    });
  }
};


} // namespace modules
