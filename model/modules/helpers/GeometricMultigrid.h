#pragma once

#include "main_header.h"
#include "coupler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace modules {


// Structured, matrix-free geometric multigrid for a pure-fluid pressure projection. The hierarchy uses physical-
// coordinate quadratic cell-centered interpolation and its volume-weighted adjoint, so a symmetric V-cycle remains
// suitable for finite-volume CG on nonuniform grids. Uniform directions remain uniform when their extents are not
// divisible by the coarsening factor. Horizontal rank agglomeration keeps coarse grids large enough for GPUs and
// removes MPI tasks as the grid shrinks.
template <class Scalar>
class GeometricMultigrid {
public:
  using Device3d = yakl::Array<Scalar ***>;
  using Device2d = yakl::Array<Scalar **>;
  using Device1d = yakl::Array<Scalar *>;
  using DeviceInt2d = yakl::Array<int **>;
  using DeviceInt1d = yakl::Array<int *>;
  using Host3d = yakl::Array<Scalar ***,Kokkos::HostSpace>;
  using Host2d = yakl::Array<Scalar **,Kokkos::HostSpace>;

  struct Block {
    int rank = -1;
    int nx = 0;
    int ny = 0;
    int ox = 0;
    int oy = 0;
    Device3d device;
    Host3d host;
  };

  struct TransferMap {
    DeviceInt2d prolong_indices;
    Device2d prolong_weights;
    DeviceInt1d restrict_offsets;
    DeviceInt1d restrict_indices;
    Device1d restrict_weights;
  };

  struct TransferHalo {
    int width = 0;
    Device3d send_lower;
    Device3d send_upper;
    Device3d recv_lower;
    Device3d recv_upper;
    Host3d send_lower_host;
    Host3d send_upper_host;
    Host3d recv_lower_host;
    Host3d recv_upper_host;
  };

  struct Transition {
    bool coarsen_x = false;
    bool coarsen_y = false;
    bool coarsen_z = false;
    bool periodic_x = false;
    bool periodic_y = false;
    bool periodic_z = false;
    bool leader = false;
    int leader_rank = -1;
    int local_nx = 0;
    int local_ny = 0;
    int local_nz = 0;
    int coarse_begin_x = 0;
    int coarse_begin_y = 0;
    int factor_x = 1;
    int factor_y = 1;
    int factor_z = 1;
    TransferMap map_x;
    TransferMap map_y;
    TransferMap map_z;
    TransferHalo restrict_x_halo;
    TransferHalo restrict_y_halo;
    TransferHalo prolong_x_halo;
    TransferHalo prolong_y_halo;
    Device3d restrict_x;
    Device3d restrict_y;
    Device3d local_coarse;
    Device3d prolong_z;
    Device3d prolong_y;
    Host3d local_host;
    std::vector<Block> blocks;
  };

  struct Level {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int nx_global = 0;
    int ny_global = 0;
    int nproc_x = 1;
    int nproc_y = 1;
    int px = 0;
    int py = 0;
    int rank = 0;
    int nranks = 1;
    int global_begin_x = 0;
    int global_begin_y = 0;
    bool periodic_x = false;
    bool periodic_y = false;
    bool periodic_z = false;
    bool owns_comm = false;
    MPI_Comm comm = MPI_COMM_NULL;
    std::vector<Scalar> dx_host;
    std::vector<Scalar> dy_host;
    std::vector<Scalar> dz_host;
    std::vector<Scalar> dx_global_host;
    std::vector<Scalar> dy_global_host;
    yakl::Array<Scalar *> x_minus;
    yakl::Array<Scalar *> x_plus;
    yakl::Array<Scalar *> y_minus;
    yakl::Array<Scalar *> y_plus;
    yakl::Array<Scalar *> z_minus;
    yakl::Array<Scalar *> z_plus;
    Device3d x;
    Device3d b;
    Device3d residual;
    Device3d x_next;
    Device2d send_west;
    Device2d send_east;
    Device2d recv_west;
    Device2d recv_east;
    Device2d send_south;
    Device2d send_north;
    Device2d recv_south;
    Device2d recv_north;
    Host2d send_west_host;
    Host2d send_east_host;
    Host2d recv_west_host;
    Host2d recv_east_host;
    Host2d send_south_host;
    Host2d send_north_host;
    Host2d recv_south_host;
    Host2d recv_north_host;
    std::unique_ptr<Transition> transition;
  };

  struct Exchange {
    std::array<MPI_Request,8> requests;
    int count = 0;
  };

  std::vector<std::unique_ptr<Level>> levels_;
  MPI_Comm root_comm_ = MPI_COMM_NULL;
  int fine_size_ = 0;
  int vcycles_ = 1;
  int pre_smooth_ = 2;
  int post_smooth_ = 2;
  int coarse_smooth_ = 24;
  Scalar jacobi_weight_ = Scalar(2)/Scalar(3);
  int coarsening_factor_ = 2;
  bool uses_mpi_transfers_ = false;
  bool initialized_ = false;


  static MPI_Datatype mpi_scalar_type() {
    if constexpr (std::is_same_v<Scalar,float>) return MPI_FLOAT;
    if constexpr (std::is_same_v<Scalar,double>) return MPI_DOUBLE;
    endrun("ERROR: unsupported geometric multigrid scalar type");
    return MPI_DATATYPE_NULL;
  }


  KOKKOS_INLINE_FUNCTION static int map_index(int index, int extent, bool periodic) {
    if (periodic) return (index%extent+extent)%extent;
    return index < 0 ? 0 : (index >= extent ? extent-1 : index);
  }


  static int coarsened_extent(int fine_extent, int factor) {
    return (fine_extent+factor-1)/factor;
  }


  static int coarsened_begin(int fine_begin, int factor) {
    return (fine_begin+factor-1)/factor;
  }


  static bool widths_are_uniform(std::vector<Scalar> const &widths) {
    if (widths.size() < 2) return true;
    Scalar total = 0;
    for (auto const width : widths) total += width;
    Scalar const mean = total/Scalar(widths.size());
    Scalar const tolerance = Scalar(64)*std::numeric_limits<Scalar>::epsilon()*
                             std::max(Scalar(1),std::abs(mean));
    for (auto const width : widths) {
      if (std::abs(width-mean) > tolerance) return false;
    }
    return true;
  }


  static std::vector<Scalar> coarsen_widths(std::vector<Scalar> const &fine, int factor) {
    if (factor == 1) return fine;
    int const coarse_extent = coarsened_extent(fine.size(),factor);
    std::vector<Scalar> coarse(coarse_extent,0);
    if (widths_are_uniform(fine)) {
      Scalar domain_length = 0;
      for (auto const width : fine) domain_length += width;
      std::fill(coarse.begin(),coarse.end(),domain_length/Scalar(coarse_extent));
      return coarse;
    }
    for (int q = 0; q < coarse.size(); q++) {
      for (int remainder = 0; remainder < factor; remainder++) {
        int const fine_index = factor*q+remainder;
        if (fine_index < fine.size()) coarse[q] += fine[fine_index];
      }
    }
    return coarse;
  }


  static std::vector<Scalar> cell_centers(std::vector<Scalar> const &widths) {
    std::vector<Scalar> centers(widths.size());
    Scalar interface = 0;
    for (int i = 0; i < widths.size(); i++) {
      centers[i] = interface+Scalar(0.5)*widths[i];
      interface += widths[i];
    }
    return centers;
  }


  static int nearest_center(std::vector<Scalar> const &centers, Scalar coordinate) {
    auto const upper = std::lower_bound(centers.begin(),centers.end(),coordinate);
    if (upper == centers.begin()) return 0;
    if (upper == centers.end()) return centers.size()-1;
    int const right = upper-centers.begin();
    int const left = right-1;
    return coordinate-centers[left] <= centers[right]-coordinate ? left : right;
  }


  static int distance_to_owned_interval(int relative, int count) {
    if (relative < 0) return -relative;
    if (relative >= count) return relative-count+1;
    return 0;
  }


  static int nearest_periodic_relative(int index, int begin, int count, int global_extent,
                                       bool periodic) {
    int relative = index-begin;
    if (!periodic) return relative;

    // Select the periodic image nearest the complete owned interval [0,count), not nearest its lower end.
    // The latter maps upper-halo points to the opposite side on uneven MPI partitions.
    int distance = distance_to_owned_interval(relative,count);
    int const plus = relative+global_extent;
    int const plus_distance = distance_to_owned_interval(plus,count);
    if (plus_distance < distance) {
      relative = plus;
      distance = plus_distance;
    }
    int const minus = index-begin-global_extent;
    int const minus_distance = distance_to_owned_interval(minus,count);
    if (minus_distance < distance) relative = minus;
    return relative;
  }


  static int local_relative(int index, int begin, int count, int global_extent, bool periodic) {
    if (begin == 0 && count == global_extent) return index;
    return nearest_periodic_relative(index,begin,count,global_extent,periodic);
  }


  static void initialize_transfer(std::vector<Scalar> const &fine_widths, int factor, bool periodic,
                                  int fine_begin, int fine_count, int coarse_begin, int coarse_count,
                                  std::string const &label, TransferMap &map,
                                  int &restrict_halo_width, int &prolong_halo_width) {
    int const nf = fine_widths.size();
    auto const coarse_widths = coarsen_widths(fine_widths,factor);
    int const nc = coarse_widths.size();
    auto const fine_centers = cell_centers(fine_widths);
    auto const coarse_centers = cell_centers(coarse_widths);
    Scalar domain_length = 0;
    for (auto const width : fine_widths) domain_length += width;

    std::vector<int> global_indices(3*nf);
    std::vector<Scalar> global_weights(3*nf,0);
    for (int i = 0; i < nf; i++) {
      int const parent = nearest_center(coarse_centers,fine_centers[i]);
      if (nc == 1) {
        global_indices[3*i] = 0;
        global_weights[3*i] = 1;
        global_indices[3*i+1] = 0;
        global_indices[3*i+2] = 0;
        continue;
      }
      Scalar coordinates[3];
      for (int entry = 0; entry < 3; entry++) {
        int const raw = parent+entry-1;
        global_indices[3*i+entry] = map_index(raw,nc,periodic);
        if (raw < 0) {
          coordinates[entry] = periodic ? coarse_centers[nc-1]-domain_length : -coarse_centers[0];
        } else if (raw >= nc) {
          coordinates[entry] = periodic ? coarse_centers[0]+domain_length :
                                          Scalar(2)*domain_length-coarse_centers[nc-1];
        } else {
          coordinates[entry] = coarse_centers[raw];
        }
      }
      for (int entry = 0; entry < 3; entry++) {
        Scalar weight = 1;
        for (int other = 0; other < 3; other++) {
          if (other != entry) {
            weight *= (fine_centers[i]-coordinates[other])/(coordinates[entry]-coordinates[other]);
          }
        }
        global_weights[3*i+entry] = weight;
      }
    }

    yakl::Array<int **,Kokkos::HostSpace> prolong_indices_host(label+"_prolong_indices_host",fine_count,3);
    yakl::Array<Scalar **,Kokkos::HostSpace> prolong_weights_host(label+"_prolong_weights_host",fine_count,3);
    prolong_halo_width = 0;
    for (int local = 0; local < fine_count; local++) {
      int const global = fine_begin+local;
      for (int entry = 0; entry < 3; entry++) {
        int const relative = local_relative(global_indices[3*global+entry],coarse_begin,coarse_count,nc,periodic);
        prolong_indices_host(local,entry) = relative;
        prolong_weights_host(local,entry) = global_weights[3*global+entry];
        prolong_halo_width = std::max(prolong_halo_width,std::max(-relative,relative-coarse_count+1));
      }
    }

    std::vector<std::vector<std::pair<int,Scalar>>> transpose(coarse_count);
    restrict_halo_width = 0;
    for (int global = 0; global < nf; global++) {
      for (int entry = 0; entry < 3; entry++) {
        Scalar const weight = global_weights[3*global+entry];
        if (weight == 0) continue;
        int const coarse_global = global_indices[3*global+entry];
        if (coarse_global < coarse_begin || coarse_global >= coarse_begin+coarse_count) continue;
        int const relative = local_relative(global,fine_begin,fine_count,nf,periodic);
        // R = Vc^{-1} P^T Vf in this direction. Tensoring these maps produces the full cell-volume adjoint.
        Scalar const restriction_weight = weight*fine_widths[global]/coarse_widths[coarse_global];
        transpose[coarse_global-coarse_begin].emplace_back(relative,restriction_weight);
        restrict_halo_width = std::max(restrict_halo_width,std::max(-relative,relative-fine_count+1));
      }
    }
    yakl::Array<int *,Kokkos::HostSpace> restrict_offsets_host(label+"_restrict_offsets_host",coarse_count+1);
    restrict_offsets_host(0) = 0;
    for (int q = 0; q < coarse_count; q++) {
      restrict_offsets_host(q+1) = restrict_offsets_host(q)+transpose[q].size();
    }
    int const num_entries = restrict_offsets_host(coarse_count);
    yakl::Array<int *,Kokkos::HostSpace> restrict_indices_host(label+"_restrict_indices_host",num_entries);
    yakl::Array<Scalar *,Kokkos::HostSpace> restrict_weights_host(label+"_restrict_weights_host",num_entries);
    int position = 0;
    for (auto const &row : transpose) {
      for (auto const &entry : row) {
        restrict_indices_host(position) = entry.first;
        restrict_weights_host(position) = entry.second;
        position++;
      }
    }
    map.prolong_indices = prolong_indices_host.createDeviceCopy();
    map.prolong_weights = prolong_weights_host.createDeviceCopy();
    map.restrict_offsets = restrict_offsets_host.createDeviceCopy();
    map.restrict_indices = restrict_indices_host.createDeviceCopy();
    map.restrict_weights = restrict_weights_host.createDeviceCopy();
  }


  // Choose integer semi-coarsening factors that make the next global grid as close to cubical in cell counts as
  // possible. Among equally isotropic choices, prefer the one that removes the most cells.
  static std::array<int,3> choose_coarsening_factors(int nx, int ny, int nz, int requested_factor) {
    std::array<int,3> best = {1,1,1};
    double best_aspect = std::numeric_limits<double>::max();
    int best_product = 0;
    for (int fz = 1; fz <= requested_factor; fz++) {
      if (fz > 1 && coarsened_extent(nz,fz) == nz) continue;
      for (int fy = 1; fy <= requested_factor; fy++) {
        if (fy > 1 && coarsened_extent(ny,fy) == ny) continue;
        for (int fx = 1; fx <= requested_factor; fx++) {
          if (fx > 1 && coarsened_extent(nx,fx) == nx) continue;
          if (fx == 1 && fy == 1 && fz == 1) continue;
          int const next_nx = coarsened_extent(nx,fx);
          int const next_ny = coarsened_extent(ny,fy);
          int const next_nz = coarsened_extent(nz,fz);
          int const minimum = std::min({next_nx,next_ny,next_nz});
          int const maximum = std::max({next_nx,next_ny,next_nz});
          double const aspect = static_cast<double>(maximum)/minimum;
          int const product = fx*fy*fz;
          if (aspect < best_aspect || (aspect == best_aspect && product > best_product)) {
            best = {fx,fy,fz};
            best_aspect = aspect;
            best_product = product;
          }
        }
      }
    }
    return best;
  }


  static void allocate_transfer_halo(TransferHalo &halo, int width, int nz, int ny, int nx,
                                     bool x_direction, std::string const &label) {
    halo.width = width;
    if (width == 0) return;
    int const halo_ny = x_direction ? ny : width;
    int const halo_nx = x_direction ? width : nx;
    halo.send_lower = Device3d(label+"_send_lower",nz,halo_ny,halo_nx);
    halo.send_upper = Device3d(label+"_send_upper",nz,halo_ny,halo_nx);
    halo.recv_lower = Device3d(label+"_recv_lower",nz,halo_ny,halo_nx);
    halo.recv_upper = Device3d(label+"_recv_upper",nz,halo_ny,halo_nx);
    halo.send_lower_host = Host3d(label+"_send_lower_host",nz,halo_ny,halo_nx);
    halo.send_upper_host = Host3d(label+"_send_upper_host",nz,halo_ny,halo_nx);
    halo.recv_lower_host = Host3d(label+"_recv_lower_host",nz,halo_ny,halo_nx);
    halo.recv_upper_host = Host3d(label+"_recv_upper_host",nz,halo_ny,halo_nx);
  }


  void allocate_level(Level &level) const {
    level.x        = Device3d("geometric_multigrid_x"       ,level.nz,level.ny,level.nx);
    level.b        = Device3d("geometric_multigrid_b"       ,level.nz,level.ny,level.nx);
    level.residual = Device3d("geometric_multigrid_residual",level.nz,level.ny,level.nx);
    level.x_next   = Device3d("geometric_multigrid_x_next"  ,level.nz,level.ny,level.nx);
    level.x_next = 0;
    level.send_west = Device2d("geometric_multigrid_send_west",level.nz,level.ny);
    level.send_east = Device2d("geometric_multigrid_send_east",level.nz,level.ny);
    level.recv_west = Device2d("geometric_multigrid_recv_west",level.nz,level.ny);
    level.recv_east = Device2d("geometric_multigrid_recv_east",level.nz,level.ny);
    level.send_south = Device2d("geometric_multigrid_send_south",level.nz,level.nx);
    level.send_north = Device2d("geometric_multigrid_send_north",level.nz,level.nx);
    level.recv_south = Device2d("geometric_multigrid_recv_south",level.nz,level.nx);
    level.recv_north = Device2d("geometric_multigrid_recv_north",level.nz,level.nx);
    level.send_west_host = Host2d("geometric_multigrid_send_west_host",level.nz,level.ny);
    level.send_east_host = Host2d("geometric_multigrid_send_east_host",level.nz,level.ny);
    level.recv_west_host = Host2d("geometric_multigrid_recv_west_host",level.nz,level.ny);
    level.recv_east_host = Host2d("geometric_multigrid_recv_east_host",level.nz,level.ny);
    level.send_south_host = Host2d("geometric_multigrid_send_south_host",level.nz,level.nx);
    level.send_north_host = Host2d("geometric_multigrid_send_north_host",level.nz,level.nx);
    level.recv_south_host = Host2d("geometric_multigrid_recv_south_host",level.nz,level.nx);
    level.recv_north_host = Host2d("geometric_multigrid_recv_north_host",level.nz,level.nx);

    yakl::Array<Scalar *,Kokkos::HostSpace> x_minus_host("geometric_multigrid_x_minus_host",level.nx);
    yakl::Array<Scalar *,Kokkos::HostSpace> x_plus_host("geometric_multigrid_x_plus_host",level.nx);
    yakl::Array<Scalar *,Kokkos::HostSpace> y_minus_host("geometric_multigrid_y_minus_host",level.ny);
    yakl::Array<Scalar *,Kokkos::HostSpace> y_plus_host("geometric_multigrid_y_plus_host",level.ny);
    std::vector<Scalar> boundary_widths(4*level.nranks);
    Scalar local_boundary_widths[4] = {level.dx_host.front(),level.dx_host.back(),
                                       level.dy_host.front(),level.dy_host.back()};
    MPI_Allgather(local_boundary_widths,4,mpi_scalar_type(),boundary_widths.data(),4,mpi_scalar_type(),level.comm);
    int const west = neighbor(level,-1,0);
    int const east = neighbor(level, 1,0);
    int const south = neighbor(level,0,-1);
    int const north = neighbor(level,0, 1);
    // Finite-volume face flux divided by the receiving cell width. Opposite entries differ on a nonuniform grid, but
    // multiplication by their cell volumes recovers the same shared-face coefficient and hence weighted symmetry.
    for (int i = 0; i < level.nx; i++) {
      x_minus_host(i) = 0;
      x_plus_host(i) = 0;
      if (i > 0 || west != MPI_PROC_NULL) {
        Scalar const adjacent = i > 0 ? level.dx_host[i-1] : boundary_widths[4*west+1];
        Scalar const distance = Scalar(0.5)*(level.dx_host[i]+adjacent);
        x_minus_host(i) = Scalar(1)/(level.dx_host[i]*distance);
      }
      if (i+1 < level.nx || east != MPI_PROC_NULL) {
        Scalar const adjacent = i+1 < level.nx ? level.dx_host[i+1] : boundary_widths[4*east];
        Scalar const distance = Scalar(0.5)*(level.dx_host[i]+adjacent);
        x_plus_host(i) = Scalar(1)/(level.dx_host[i]*distance);
      }
    }
    for (int j = 0; j < level.ny; j++) {
      y_minus_host(j) = 0;
      y_plus_host(j) = 0;
      if (j > 0 || south != MPI_PROC_NULL) {
        Scalar const adjacent = j > 0 ? level.dy_host[j-1] : boundary_widths[4*south+3];
        Scalar const distance = Scalar(0.5)*(level.dy_host[j]+adjacent);
        y_minus_host(j) = Scalar(1)/(level.dy_host[j]*distance);
      }
      if (j+1 < level.ny || north != MPI_PROC_NULL) {
        Scalar const adjacent = j+1 < level.ny ? level.dy_host[j+1] : boundary_widths[4*north+2];
        Scalar const distance = Scalar(0.5)*(level.dy_host[j]+adjacent);
        y_plus_host(j) = Scalar(1)/(level.dy_host[j]*distance);
      }
    }
    level.x_minus = x_minus_host.createDeviceCopy();
    level.x_plus = x_plus_host.createDeviceCopy();
    level.y_minus = y_minus_host.createDeviceCopy();
    level.y_plus = y_plus_host.createDeviceCopy();

    yakl::Array<Scalar *,Kokkos::HostSpace> z_minus_host("geometric_multigrid_z_minus_host",level.nz);
    yakl::Array<Scalar *,Kokkos::HostSpace> z_plus_host("geometric_multigrid_z_plus_host",level.nz);
    for (int k = 0; k < level.nz; k++) {
      z_minus_host(k) = 0;
      z_plus_host(k) = 0;
      if (k > 0 || level.periodic_z) {
        int const km = k > 0 ? k-1 : level.nz-1;
        Scalar const distance = Scalar(0.5)*(level.dz_host[k]+level.dz_host[km]);
        z_minus_host(k) = Scalar(1)/(level.dz_host[k]*distance);
      }
      if (k+1 < level.nz || level.periodic_z) {
        int const kp = k+1 < level.nz ? k+1 : 0;
        Scalar const distance = Scalar(0.5)*(level.dz_host[k]+level.dz_host[kp]);
        z_plus_host(k) = Scalar(1)/(level.dz_host[k]*distance);
      }
    }
    level.z_minus = z_minus_host.createDeviceCopy();
    level.z_plus = z_plus_host.createDeviceCopy();
    level.x = 0;
    level.b = 0;
  }


  static int neighbor(Level const &level, int dx, int dy) {
    int px = level.px+dx;
    int py = level.py+dy;
    if (px < 0 || px >= level.nproc_x) {
      if (!level.periodic_x) return MPI_PROC_NULL;
      px = (px+level.nproc_x)%level.nproc_x;
    }
    if (py < 0 || py >= level.nproc_y) {
      if (!level.periodic_y) return MPI_PROC_NULL;
      py = (py+level.nproc_y)%level.nproc_y;
    }
    return py*level.nproc_x+px;
  }


  static void post_receive(Exchange &exchange, void *data, int count, int source, int tag, MPI_Datatype type,
                           MPI_Comm comm) {
    if (source == MPI_PROC_NULL) return;
    MPI_Irecv(data,count,type,source,tag,comm,&exchange.requests[exchange.count++]);
  }


  static void post_send(Exchange &exchange, void *data, int count, int destination, int tag, MPI_Datatype type,
                        MPI_Comm comm) {
    if (destination == MPI_PROC_NULL) return;
    MPI_Isend(data,count,type,destination,tag,comm,&exchange.requests[exchange.count++]);
  }


  static void exchange_transfer_halo(Level const &level, Device3d const &input, TransferHalo &halo,
                                     bool x_direction, int tag) {
    int const width = halo.width;
    if (width == 0) return;
    int const nx = input.extent(2);
    int const ny = input.extent(1);
    int const nz = input.extent(0);
    int const local_extent = x_direction ? nx : ny;
    if (width > local_extent) {
      endrun("ERROR: geometric multigrid transfer halo is wider than the local MPI subdomain");
    }
    auto send_lower = halo.send_lower;
    auto send_upper = halo.send_upper;
    if (x_direction) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,width),
                         KOKKOS_LAMBDA (int k, int j, int i) {
        send_lower(k,j,i) = input(k,j,i);
        send_upper(k,j,i) = input(k,j,nx-width+i);
      });
    } else {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,width,nx),
                         KOKKOS_LAMBDA (int k, int j, int i) {
        send_lower(k,j,i) = input(k,j,i);
        send_upper(k,j,i) = input(k,ny-width+j,i);
      });
    }
    int const lower = x_direction ? neighbor(level,-1,0) : neighbor(level,0,-1);
    int const upper = x_direction ? neighbor(level, 1,0) : neighbor(level,0, 1);
    Exchange exchange;
    #ifdef PORTURB_GPU_AWARE_MPI
      Kokkos::fence();
      post_receive(exchange,halo.recv_lower.data(),halo.recv_lower.size(),lower,tag+1,mpi_scalar_type(),level.comm);
      post_receive(exchange,halo.recv_upper.data(),halo.recv_upper.size(),upper,tag,mpi_scalar_type(),level.comm);
      post_send(exchange,halo.send_lower.data(),halo.send_lower.size(),lower,tag,mpi_scalar_type(),level.comm);
      post_send(exchange,halo.send_upper.data(),halo.send_upper.size(),upper,tag+1,mpi_scalar_type(),level.comm);
    #else
      halo.send_lower.deep_copy_to(halo.send_lower_host);
      halo.send_upper.deep_copy_to(halo.send_upper_host);
      post_receive(exchange,halo.recv_lower_host.data(),halo.recv_lower_host.size(),lower,tag+1,
                   mpi_scalar_type(),level.comm);
      post_receive(exchange,halo.recv_upper_host.data(),halo.recv_upper_host.size(),upper,tag,
                   mpi_scalar_type(),level.comm);
      post_send(exchange,halo.send_lower_host.data(),halo.send_lower_host.size(),lower,tag,
                mpi_scalar_type(),level.comm);
      post_send(exchange,halo.send_upper_host.data(),halo.send_upper_host.size(),upper,tag+1,
                mpi_scalar_type(),level.comm);
    #endif
    if (exchange.count > 0) MPI_Waitall(exchange.count,exchange.requests.data(),MPI_STATUSES_IGNORE);
    #ifndef PORTURB_GPU_AWARE_MPI
      halo.recv_lower_host.deep_copy_to(halo.recv_lower);
      halo.recv_upper_host.deep_copy_to(halo.recv_upper);
    #endif
  }


  KOKKOS_INLINE_FUNCTION static Scalar transfer_value_x(Device3d const &input, Device3d const &recv_lower,
                                                         Device3d const &recv_upper, int width,
                                                         int k, int j, int i) {
    if (i < 0) return recv_lower(k,j,width+i);
    if (i >= input.extent(2)) return recv_upper(k,j,i-input.extent(2));
    return input(k,j,i);
  }


  KOKKOS_INLINE_FUNCTION static Scalar transfer_value_y(Device3d const &input, Device3d const &recv_lower,
                                                         Device3d const &recv_upper, int width,
                                                         int k, int j, int i) {
    if (j < 0) return recv_lower(k,width+j,i);
    if (j >= input.extent(1)) return recv_upper(k,j-input.extent(1),i);
    return input(k,j,i);
  }


public:
  // NVCC requires member functions enclosing extended device lambdas to be publicly nameable.
  Exchange begin_exchange(Level &level, int level_index) const {
    auto x = level.x;
    auto send_west = level.send_west;
    auto send_east = level.send_east;
    auto send_south = level.send_south;
    auto send_north = level.send_north;
    int const nx = level.nx;
    int const ny = level.ny;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_halo_pack");
    if (level.nproc_x > 1) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<2>(level.nz,level.ny),
                         KOKKOS_LAMBDA (int k, int j) {
        send_west(k,j) = x(k,j,0);
        send_east(k,j) = x(k,j,nx-1);
      });
    }
    if (level.nproc_y > 1) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<2>(level.nz,level.nx),
                         KOKKOS_LAMBDA (int k, int i) {
        send_south(k,i) = x(k,0,i);
        send_north(k,i) = x(k,ny-1,i);
      });
    }
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_halo_pack");

    int const west = level.nproc_x > 1 ? neighbor(level,-1,0) : MPI_PROC_NULL;
    int const east = level.nproc_x > 1 ? neighbor(level, 1,0) : MPI_PROC_NULL;
    int const south = level.nproc_y > 1 ? neighbor(level,0,-1) : MPI_PROC_NULL;
    int const north = level.nproc_y > 1 ? neighbor(level,0, 1) : MPI_PROC_NULL;
    int const tag = 100+4*level_index;
    Exchange exchange;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_halo_post");
    #ifdef PORTURB_GPU_AWARE_MPI
      Kokkos::fence();
      post_receive(exchange,level.recv_west.data(),level.recv_west.size(),west,tag+1,mpi_scalar_type(),level.comm);
      post_receive(exchange,level.recv_east.data(),level.recv_east.size(),east,tag  ,mpi_scalar_type(),level.comm);
      post_receive(exchange,level.recv_south.data(),level.recv_south.size(),south,tag+3,mpi_scalar_type(),level.comm);
      post_receive(exchange,level.recv_north.data(),level.recv_north.size(),north,tag+2,mpi_scalar_type(),level.comm);
      post_send(exchange,level.send_west.data(),level.send_west.size(),west,tag  ,mpi_scalar_type(),level.comm);
      post_send(exchange,level.send_east.data(),level.send_east.size(),east,tag+1,mpi_scalar_type(),level.comm);
      post_send(exchange,level.send_south.data(),level.send_south.size(),south,tag+2,mpi_scalar_type(),level.comm);
      post_send(exchange,level.send_north.data(),level.send_north.size(),north,tag+3,mpi_scalar_type(),level.comm);
    #else
      if (level.nproc_x > 1) {
        level.send_west.deep_copy_to(level.send_west_host);
        level.send_east.deep_copy_to(level.send_east_host);
      }
      if (level.nproc_y > 1) {
        level.send_south.deep_copy_to(level.send_south_host);
        level.send_north.deep_copy_to(level.send_north_host);
      }
      post_receive(exchange,level.recv_west_host.data(),level.recv_west_host.size(),west,tag+1,
                   mpi_scalar_type(),level.comm);
      post_receive(exchange,level.recv_east_host.data(),level.recv_east_host.size(),east,tag,
                   mpi_scalar_type(),level.comm);
      post_receive(exchange,level.recv_south_host.data(),level.recv_south_host.size(),south,tag+3,
                   mpi_scalar_type(),level.comm);
      post_receive(exchange,level.recv_north_host.data(),level.recv_north_host.size(),north,tag+2,
                   mpi_scalar_type(),level.comm);
      post_send(exchange,level.send_west_host.data(),level.send_west_host.size(),west,tag,
                mpi_scalar_type(),level.comm);
      post_send(exchange,level.send_east_host.data(),level.send_east_host.size(),east,tag+1,
                mpi_scalar_type(),level.comm);
      post_send(exchange,level.send_south_host.data(),level.send_south_host.size(),south,tag+2,
                mpi_scalar_type(),level.comm);
      post_send(exchange,level.send_north_host.data(),level.send_north_host.size(),north,tag+3,
                mpi_scalar_type(),level.comm);
    #endif
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_halo_post");
    return exchange;
  }


  static void finish_exchange(Level &level, Exchange &exchange) {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_halo_wait");
    if (exchange.count > 0) MPI_Waitall(exchange.count,exchange.requests.data(),MPI_STATUSES_IGNORE);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_halo_wait");
    #ifndef PORTURB_GPU_AWARE_MPI
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_halo_receive_stage");
      if (level.nproc_x > 1) {
        level.recv_west_host.deep_copy_to(level.recv_west);
        level.recv_east_host.deep_copy_to(level.recv_east);
      }
      if (level.nproc_y > 1) {
        level.recv_south_host.deep_copy_to(level.recv_south);
        level.recv_north_host.deep_copy_to(level.recv_north);
      }
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_halo_receive_stage");
    #endif
  }


  KOKKOS_INLINE_FUNCTION static Scalar apply_cell(Device3d const &x,
                                                   Device2d const &recv_west, Device2d const &recv_east,
                                                   Device2d const &recv_south, Device2d const &recv_north,
                                                   yakl::Array<Scalar *> const &z_minus,
                                                   yakl::Array<Scalar *> const &z_plus,
                                                   yakl::Array<Scalar *> const &x_minus,
                                                   yakl::Array<Scalar *> const &x_plus,
                                                   yakl::Array<Scalar *> const &y_minus,
                                                   yakl::Array<Scalar *> const &y_plus,
                                                   int nx, int ny, int nz, int nproc_x, int nproc_y,
                                                   int px, int py, bool periodic_x, bool periodic_y,
                                                   int k, int j, int i, Scalar shift) {
    Scalar const center = x(k,j,i);
    Scalar value = 0;
    Scalar diagonal = shift;
    Scalar const wxm = x_minus(i);
    Scalar const wxp = x_plus(i);
    Scalar const wym = y_minus(j);
    Scalar const wyp = y_plus(j);
    if (i > 0) {
      value -= wxm*x(k,j,i-1);
      diagonal += wxm;
    } else if (nproc_x > 1 && (periodic_x || px > 0)) {
      value -= wxm*recv_west(k,j);
      diagonal += wxm;
    } else if (periodic_x) {
      value -= wxm*x(k,j,nx-1);
      diagonal += wxm;
    }
    if (i+1 < nx) {
      value -= wxp*x(k,j,i+1);
      diagonal += wxp;
    } else if (nproc_x > 1 && (periodic_x || px+1 < nproc_x)) {
      value -= wxp*recv_east(k,j);
      diagonal += wxp;
    } else if (periodic_x) {
      value -= wxp*x(k,j,0);
      diagonal += wxp;
    }
    if (j > 0) {
      value -= wym*x(k,j-1,i);
      diagonal += wym;
    } else if (nproc_y > 1 && (periodic_y || py > 0)) {
      value -= wym*recv_south(k,i);
      diagonal += wym;
    } else if (periodic_y) {
      value -= wym*x(k,ny-1,i);
      diagonal += wym;
    }
    if (j+1 < ny) {
      value -= wyp*x(k,j+1,i);
      diagonal += wyp;
    } else if (nproc_y > 1 && (periodic_y || py+1 < nproc_y)) {
      value -= wyp*recv_north(k,i);
      diagonal += wyp;
    } else if (periodic_y) {
      value -= wyp*x(k,0,i);
      diagonal += wyp;
    }
    if (z_minus(k) > 0) {
      int const km = k > 0 ? k-1 : nz-1;
      value -= z_minus(k)*x(km,j,i);
      diagonal += z_minus(k);
    }
    if (z_plus(k) > 0) {
      int const kp = k+1 < nz ? k+1 : 0;
      value -= z_plus(k)*x(kp,j,i);
      diagonal += z_plus(k);
    }
    return value+diagonal*center;
  }


  void smooth(Level &level, int iterations, Scalar shift, int level_index,
              bool zero_initial_guess = false) const {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_jacobi_smooth");
    auto b = level.b;
    auto recv_west = level.recv_west;
    auto recv_east = level.recv_east;
    auto recv_south = level.recv_south;
    auto recv_north = level.recv_north;
    auto z_minus = level.z_minus;
    auto z_plus = level.z_plus;
    auto x_minus = level.x_minus;
    auto x_plus = level.x_plus;
    auto y_minus = level.y_minus;
    auto y_plus = level.y_plus;
    Scalar const weight = jacobi_weight_;
    int const nx = level.nx;
    int const ny = level.ny;
    int const nz = level.nz;
    int const nproc_x = level.nproc_x;
    int const nproc_y = level.nproc_y;
    int const px = level.px;
    int const py = level.py;
    bool const periodic_x = level.periodic_x;
    bool const periodic_y = level.periodic_y;
    for (int iteration = 0; iteration < iterations; iteration++) {
      auto x = level.x;
      auto x_next = level.x_next;
      bool const zero_guess = zero_initial_guess && iteration == 0;
      auto update = KOKKOS_LAMBDA (int k, int j, int i) {
        Scalar diagonal = shift+z_minus(k)+z_plus(k);
        diagonal += x_minus(i)+x_plus(i)+y_minus(j)+y_plus(j);
        if (diagonal > std::numeric_limits<Scalar>::min()) {
          if (zero_guess) {
            x_next(k,j,i) = weight*b(k,j,i)/diagonal;
          } else {
            Scalar const Ax = apply_cell(x,recv_west,recv_east,recv_south,recv_north,z_minus,z_plus,
                                         x_minus,x_plus,y_minus,y_plus,nx,ny,nz,nproc_x,nproc_y,px,py,
                                         periodic_x,periodic_y,k,j,i,shift);
            x_next(k,j,i) = x(k,j,i)+weight*(b(k,j,i)-Ax)/diagonal;
          }
        } else {
          x_next(k,j,i) = 0;
        }
      };
      if (zero_guess || level.nranks == 1) {
        yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,nx),update);
      } else {
        Exchange exchange = begin_exchange(level,level_index);
        if (nx > 2 && ny > 2) {
          yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny-2,nx-2),
                             KOKKOS_LAMBDA (int k, int jj, int ii) { update(k,jj+1,ii+1); });
        }
        finish_exchange(level,exchange);
        yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,nx),
                           KOKKOS_LAMBDA (int k, int j, int i) {
          if (i > 0 && i+1 < nx && j > 0 && j+1 < ny) return;
          update(k,j,i);
        });
      }
      level.x = x_next;
      level.x_next = x;
    }
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_jacobi_smooth");
  }


  void compute_residual(Level &level, Scalar shift, int level_index) const {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_residual");
    auto x = level.x;
    auto b = level.b;
    auto residual = level.residual;
    auto recv_west = level.recv_west;
    auto recv_east = level.recv_east;
    auto recv_south = level.recv_south;
    auto recv_north = level.recv_north;
    auto z_minus = level.z_minus;
    auto z_plus = level.z_plus;
    auto x_minus = level.x_minus;
    auto x_plus = level.x_plus;
    auto y_minus = level.y_minus;
    auto y_plus = level.y_plus;
    int const nx = level.nx;
    int const ny = level.ny;
    int const nz = level.nz;
    int const nproc_x = level.nproc_x;
    int const nproc_y = level.nproc_y;
    int const px = level.px;
    int const py = level.py;
    bool const periodic_x = level.periodic_x;
    bool const periodic_y = level.periodic_y;
    auto calculate = KOKKOS_LAMBDA (int k, int j, int i) {
      residual(k,j,i) = b(k,j,i)-apply_cell(x,recv_west,recv_east,recv_south,recv_north,z_minus,z_plus,
                                            x_minus,x_plus,y_minus,y_plus,nx,ny,nz,nproc_x,nproc_y,px,py,
                                            periodic_x,periodic_y,k,j,i,shift);
    };
    if (level.nranks == 1) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,nx),calculate);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_residual");
      return;
    }
    Exchange exchange = begin_exchange(level,level_index);
    if (nx > 2 && ny > 2) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny-2,nx-2),
                         KOKKOS_LAMBDA (int k, int jj, int ii) { calculate(k,jj+1,ii+1); });
    }
    finish_exchange(level,exchange);
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      if (i > 0 && i+1 < nx && j > 0 && j+1 < ny) return;
      calculate(k,j,i);
    });
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_residual");
  }


  void apply_smoother(Level &level, int iterations, Scalar shift, int level_index,
                      bool zero_initial_guess = false) const {
    if (zero_initial_guess && iterations == 0) {
      level.x = 0;
      return;
    }
    smooth(level,iterations,shift,level_index,zero_initial_guess);
  }


  static void restrict_dimension_x(Level const &fine, Transition &transition, int level_index) {
    auto input = fine.residual;
    auto output = transition.restrict_x;
    if (!transition.coarsen_x) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,fine.nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) = input(k,j,i); });
      return;
    }
    if (fine.nproc_x > 1) exchange_transfer_halo(fine,input,transition.restrict_x_halo,true,2000+8*level_index);
    auto offsets = transition.map_x.restrict_offsets;
    auto indices = transition.map_x.restrict_indices;
    auto weights = transition.map_x.restrict_weights;
    auto recv_lower = transition.restrict_x_halo.recv_lower;
    auto recv_upper = transition.restrict_x_halo.recv_upper;
    int const width = transition.restrict_x_halo.width;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,transition.local_nx),
                       KOKKOS_LAMBDA (int k, int j, int q) {
      Scalar value = 0;
      for (int entry = offsets(q); entry < offsets(q+1); entry++) {
        value += weights(entry)*transfer_value_x(input,recv_lower,recv_upper,width,k,j,indices(entry));
      }
      output(k,j,q) = value;
    });
  }


  static void restrict_dimension_y(Level const &fine, Transition &transition, int level_index) {
    auto input = transition.restrict_x;
    auto output = transition.restrict_y;
    int const nx = transition.local_nx;
    if (!transition.coarsen_y) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) = input(k,j,i); });
      return;
    }
    if (fine.nproc_y > 1) exchange_transfer_halo(fine,input,transition.restrict_y_halo,false,2002+8*level_index);
    auto offsets = transition.map_y.restrict_offsets;
    auto indices = transition.map_y.restrict_indices;
    auto weights = transition.map_y.restrict_weights;
    auto recv_lower = transition.restrict_y_halo.recv_lower;
    auto recv_upper = transition.restrict_y_halo.recv_upper;
    int const width = transition.restrict_y_halo.width;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,transition.local_ny,nx),
                       KOKKOS_LAMBDA (int k, int q, int i) {
      Scalar value = 0;
      for (int entry = offsets(q); entry < offsets(q+1); entry++) {
        value += weights(entry)*transfer_value_y(input,recv_lower,recv_upper,width,k,indices(entry),i);
      }
      output(k,q,i) = value;
    });
  }


  static void restrict_dimension_z(Level const &fine, Transition &transition) {
    auto input = transition.restrict_y;
    auto output = transition.local_coarse;
    int const nx = transition.local_nx;
    int const ny = transition.local_ny;
    if (!transition.coarsen_z) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,ny,nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) = input(k,j,i); });
      return;
    }
    auto offsets = transition.map_z.restrict_offsets;
    auto indices = transition.map_z.restrict_indices;
    auto weights = transition.map_z.restrict_weights;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(transition.local_nz,ny,nx),
                       KOKKOS_LAMBDA (int q, int j, int i) {
      Scalar value = 0;
      for (int entry = offsets(q); entry < offsets(q+1); entry++) {
        value += weights(entry)*input(indices(entry),j,i);
      }
      output(q,j,i) = value;
    });
  }


  static void prolong_dimension_z(Level const &fine, Transition &transition) {
    auto input = transition.local_coarse;
    auto output = transition.prolong_z;
    int const nx = transition.local_nx;
    int const ny = transition.local_ny;
    if (!transition.coarsen_z) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,ny,nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) = input(k,j,i); });
      return;
    }
    auto indices = transition.map_z.prolong_indices;
    auto weights = transition.map_z.prolong_weights;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,ny,nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      Scalar value = 0;
      for (int entry = 0; entry < 3; entry++) {
        value += weights(k,entry)*input(indices(k,entry),j,i);
      }
      output(k,j,i) = value;
    });
  }


  static void prolong_dimension_y(Level const &fine, Transition &transition, int level_index) {
    auto input = transition.prolong_z;
    auto output = transition.prolong_y;
    int const nx = transition.local_nx;
    if (!transition.coarsen_y) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) = input(k,j,i); });
      return;
    }
    if (fine.nproc_y > 1) exchange_transfer_halo(fine,input,transition.prolong_y_halo,false,2004+8*level_index);
    auto indices = transition.map_y.prolong_indices;
    auto weights = transition.map_y.prolong_weights;
    auto recv_lower = transition.prolong_y_halo.recv_lower;
    auto recv_upper = transition.prolong_y_halo.recv_upper;
    int const width = transition.prolong_y_halo.width;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      Scalar value = 0;
      for (int entry = 0; entry < 3; entry++) {
        value += weights(j,entry)*transfer_value_y(input,recv_lower,recv_upper,width,k,indices(j,entry),i);
      }
      output(k,j,i) = value;
    });
  }


  static void prolong_dimension_x(Level &fine, Transition &transition, int level_index) {
    auto input = transition.prolong_y;
    auto output = fine.x;
    if (!transition.coarsen_x) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,fine.nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) += input(k,j,i); });
      return;
    }
    if (fine.nproc_x > 1) exchange_transfer_halo(fine,input,transition.prolong_x_halo,true,2006+8*level_index);
    auto indices = transition.map_x.prolong_indices;
    auto weights = transition.map_x.prolong_weights;
    auto recv_lower = transition.prolong_x_halo.recv_lower;
    auto recv_upper = transition.prolong_x_halo.recv_upper;
    int const width = transition.prolong_x_halo.width;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,fine.nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      Scalar value = 0;
      for (int entry = 0; entry < 3; entry++) {
        value += weights(i,entry)*transfer_value_x(input,recv_lower,recv_upper,width,k,j,indices(i,entry));
      }
      output(k,j,i) += value;
    });
  }


  static void copy_block_to_level(Device3d const &block, Device3d const &level, int ox, int oy) {
    int const nz = block.extent(0);
    int const ny = block.extent(1);
    int const nx = block.extent(2);
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,nx),
                       KOKKOS_LAMBDA (int k, int j, int i) { level(k,oy+j,ox+i) = block(k,j,i); });
  }


  static void copy_level_to_block(Device3d const &level, Device3d const &block, int ox, int oy) {
    int const nz = block.extent(0);
    int const ny = block.extent(1);
    int const nx = block.extent(2);
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nz,ny,nx),
                       KOKKOS_LAMBDA (int k, int j, int i) { block(k,j,i) = level(k,oy+j,ox+i); });
  }


  void gather_restricted(Level &fine, Level *coarse, int level_index) const {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_gather");
    Transition &transition = *fine.transition;
    int const tag = 1000+level_index;
    #ifdef PORTURB_GPU_AWARE_MPI
      Kokkos::fence();
      if (!transition.leader) {
        if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_gather_mpi");
        MPI_Send(transition.local_coarse.data(),transition.local_coarse.size(),mpi_scalar_type(),
                 transition.leader_rank,tag,fine.comm);
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_gather_mpi");
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_gather");
        return;
      }
      std::vector<MPI_Request> requests;
      for (auto &block : transition.blocks) {
        if (block.rank == fine.rank) {
          copy_block_to_level(transition.local_coarse,coarse->b,block.ox,block.oy);
        } else {
          requests.emplace_back();
          MPI_Irecv(block.device.data(),block.device.size(),mpi_scalar_type(),block.rank,tag,fine.comm,&requests.back());
        }
      }
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_gather_mpi");
      if (!requests.empty()) MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_gather_mpi");
      for (auto &block : transition.blocks) {
        if (block.rank != fine.rank) copy_block_to_level(block.device,coarse->b,block.ox,block.oy);
      }
    #else
      transition.local_coarse.deep_copy_to(transition.local_host);
      if (!transition.leader) {
        if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_gather_mpi");
        MPI_Send(transition.local_host.data(),transition.local_host.size(),mpi_scalar_type(),
                 transition.leader_rank,tag,fine.comm);
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_gather_mpi");
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_gather");
        return;
      }
      std::vector<MPI_Request> requests;
      for (auto &block : transition.blocks) {
        if (block.rank == fine.rank) {
          copy_block_to_level(transition.local_coarse,coarse->b,block.ox,block.oy);
        } else {
          requests.emplace_back();
          MPI_Irecv(block.host.data(),block.host.size(),mpi_scalar_type(),block.rank,tag,fine.comm,&requests.back());
        }
      }
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_gather_mpi");
      if (!requests.empty()) MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_gather_mpi");
      for (auto &block : transition.blocks) {
        if (block.rank != fine.rank) {
          block.host.deep_copy_to(block.device);
          copy_block_to_level(block.device,coarse->b,block.ox,block.oy);
        }
      }
    #endif
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_gather");
  }


  void scatter_correction(Level &fine, Level *coarse, int level_index) const {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_scatter");
    Transition &transition = *fine.transition;
    int const tag = 2000+level_index;
    #ifdef PORTURB_GPU_AWARE_MPI
      if (!transition.leader) {
        if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_scatter_mpi");
        MPI_Recv(transition.local_coarse.data(),transition.local_coarse.size(),mpi_scalar_type(),
                 transition.leader_rank,tag,fine.comm,MPI_STATUS_IGNORE);
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_scatter_mpi");
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_scatter");
        return;
      }
      std::vector<MPI_Request> requests;
      for (auto &block : transition.blocks) {
        copy_level_to_block(coarse->x,block.rank == fine.rank ? transition.local_coarse : block.device,
                            block.ox,block.oy);
      }
      Kokkos::fence();
      for (auto &block : transition.blocks) {
        if (block.rank == fine.rank) continue;
        requests.emplace_back();
        MPI_Isend(block.device.data(),block.device.size(),mpi_scalar_type(),block.rank,tag,fine.comm,&requests.back());
      }
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_scatter_mpi");
      if (!requests.empty()) MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_scatter_mpi");
    #else
      if (!transition.leader) {
        if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_scatter_mpi");
        MPI_Recv(transition.local_host.data(),transition.local_host.size(),mpi_scalar_type(),
                 transition.leader_rank,tag,fine.comm,MPI_STATUS_IGNORE);
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_scatter_mpi");
        transition.local_host.deep_copy_to(transition.local_coarse);
        if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_scatter");
        return;
      }
      std::vector<MPI_Request> requests;
      for (auto &block : transition.blocks) {
        Device3d destination = block.rank == fine.rank ? transition.local_coarse : block.device;
        copy_level_to_block(coarse->x,destination,block.ox,block.oy);
      }
      Kokkos::fence();
      for (auto &block : transition.blocks) {
        if (block.rank == fine.rank) continue;
        block.device.deep_copy_to(block.host);
        requests.emplace_back();
        MPI_Isend(block.host.data(),block.host.size(),mpi_scalar_type(),block.rank,tag,fine.comm,&requests.back());
      }
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_aggregation_scatter_mpi");
      if (!requests.empty()) MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_scatter_mpi");
    #endif
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_aggregation_scatter");
  }


  void vcycle(int level_index, Scalar shift, bool zero_initial_guess = false) const {
    Level &level = *levels_[level_index];
    if (!level.transition) {
      if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_coarse_solve");
      apply_smoother(level,coarse_smooth_,shift,level_index,zero_initial_guess);
      if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_coarse_solve");
      return;
    }
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_pre_smooth");
    apply_smoother(level,pre_smooth_,shift,level_index,zero_initial_guess);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_pre_smooth");
    compute_residual(level,shift,level_index);
    Transition &transition = *level.transition;
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_restriction");
    restrict_dimension_x(level,transition,level_index);
    restrict_dimension_y(level,transition,level_index);
    if (transition.coarsen_z) restrict_dimension_z(level,transition);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_restriction");
    Level *coarse = transition.leader ? levels_[level_index+1].get() : nullptr;
    gather_restricted(level,coarse,level_index);
    if (transition.leader) {
      vcycle(level_index+1,shift,true);
    }
    scatter_correction(level,coarse,level_index);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_prolongation");
    if (transition.coarsen_z) prolong_dimension_z(level,transition);
    prolong_dimension_y(level,transition,level_index);
    prolong_dimension_x(level,transition,level_index);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_prolongation");
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_post_smooth");
    apply_smoother(level,post_smooth_,shift,level_index);
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_post_smooth");
  }


public:
  struct Options {
    int vcycles = 1;
    int pre_smooth = 2;
    int post_smooth = 2;
    int coarse_smooth = 24;
    int coarse_cells = 32768;
    int min_cells_per_rank = 131072;
    Scalar jacobi_weight = Scalar(2)/Scalar(3);
    int coarsening_factor = 2;
  };


  ~GeometricMultigrid() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) return;
    for (auto &level : levels_) {
      if (level->owns_comm && level->comm != MPI_COMM_NULL) MPI_Comm_free(&level->comm);
    }
  }


  bool initialized() const { return initialized_; }


  void initialize(core::Coupler &coupler, Options const &options) {
    if (initialized_) endrun("ERROR: geometric multigrid initialized more than once");
    if (options.vcycles <= 0 || options.pre_smooth < 0 || options.post_smooth < 0 ||
        options.pre_smooth != options.post_smooth || options.coarse_smooth <= 0 ||
        options.coarse_cells <= 0 || options.min_cells_per_rank <= 0 ||
        !(options.jacobi_weight > 0 && options.jacobi_weight < 1)) {
      endrun("ERROR: invalid geometric multigrid options; CG requires equal pre/post smoothing counts");
    }
    if (options.coarsening_factor < 2) endrun("ERROR: geometric multigrid coarsening factor must be at least two");
    long long const total_cells = static_cast<long long>(coupler.get_nx_glob())*coupler.get_ny_glob()*coupler.get_nz();
    int const fluid_cells = coupler.get_option<int>("dycore_anelastic_fluid_count");
    if (fluid_cells != total_cells) {
      endrun("ERROR: GeometricMultigrid requires a pure-fluid domain without immersed material");
    }
    auto valid_boundary_pair = [&] (std::string const &lower, std::string const &upper) {
      bool const periodic = lower == "periodic" && upper == "periodic";
      bool const walls = lower == "wall_free_slip" && upper == "wall_free_slip";
      return periodic || walls;
    };
    std::string const bc_x1 = coupler.get_option<std::string>("bc_x1");
    std::string const bc_x2 = coupler.get_option<std::string>("bc_x2");
    std::string const bc_y1 = coupler.get_option<std::string>("bc_y1");
    std::string const bc_y2 = coupler.get_option<std::string>("bc_y2");
    std::string const bc_z1 = coupler.get_option<std::string>("bc_z1");
    std::string const bc_z2 = coupler.get_option<std::string>("bc_z2");
    if (!valid_boundary_pair(bc_x1,bc_x2) || !valid_boundary_pair(bc_y1,bc_y2) ||
        !valid_boundary_pair(bc_z1,bc_z2)) {
      endrun("ERROR: GeometricMultigrid supports paired periodic or wall_free_slip boundaries");
    }

    root_comm_ = coupler.get_parallel_comm().get_mpi_comm();
    fine_size_ = coupler.get_nx()*coupler.get_ny()*coupler.get_nz();
    vcycles_ = options.vcycles;
    pre_smooth_ = options.pre_smooth;
    post_smooth_ = options.post_smooth;
    coarse_smooth_ = options.coarse_smooth;
    jacobi_weight_ = options.jacobi_weight;
    coarsening_factor_ = options.coarsening_factor;
    auto dz_host_array = coupler.get_dz().createHostCopy();
    std::vector<Scalar> dz(coupler.get_nz());
    for (int k = 0; k < dz.size(); k++) dz[k] = static_cast<Scalar>(dz_host_array(k));

    auto fine = std::make_unique<Level>();
    fine->nx = coupler.get_nx();
    fine->ny = coupler.get_ny();
    fine->nz = coupler.get_nz();
    fine->nx_global = coupler.get_nx_glob();
    fine->ny_global = coupler.get_ny_glob();
    fine->nproc_x = coupler.get_nproc_x();
    fine->nproc_y = coupler.get_nproc_y();
    fine->px = coupler.get_px();
    fine->py = coupler.get_py();
    fine->comm = root_comm_;
    fine->rank = coupler.get_myrank();
    fine->nranks = coupler.get_nranks();
    fine->global_begin_x = coupler.get_i_beg();
    fine->global_begin_y = coupler.get_j_beg();
    fine->periodic_x = bc_x1 == "periodic";
    fine->periodic_y = bc_y1 == "periodic";
    fine->periodic_z = bc_z1 == "periodic";
    fine->dx_host.assign(fine->nx,static_cast<Scalar>(coupler.get_dx()));
    fine->dy_host.assign(fine->ny,static_cast<Scalar>(coupler.get_dy()));
    fine->dz_host = dz;
    fine->dx_global_host.assign(fine->nx_global,static_cast<Scalar>(coupler.get_dx()));
    fine->dy_global_host.assign(fine->ny_global,static_cast<Scalar>(coupler.get_dy()));
    allocate_level(*fine);
    levels_.push_back(std::move(fine));

    bool active = true;
    int level_index = 0;
    while (active) {
      Level &level = *levels_.back();
      long long const global_cells = static_cast<long long>(level.nx_global)*level.ny_global*level.nz;
      if (level.nranks == 1 && global_cells <= options.coarse_cells) break;

      bool const geometric_target = global_cells <= options.coarse_cells;
      auto const factors = geometric_target ? std::array<int,3>{1,1,1} :
                                              choose_coarsening_factors(level.nx_global,level.ny_global,level.nz,
                                                                        coarsening_factor_);
      bool const coarsen_x = factors[0] > 1;
      bool const coarsen_y = factors[1] > 1;
      bool const coarsen_z = factors[2] > 1;
      uses_mpi_transfers_ = uses_mpi_transfers_ ||
                            (coarsen_x && level.nproc_x > 1) ||
                            (coarsen_y && level.nproc_y > 1);
      if (!coarsen_x && !coarsen_y && !coarsen_z && level.nranks == 1) break;

      int const factor_x = factors[0];
      int const factor_y = factors[1];
      int const factor_z = factors[2];
      int const coarse_begin_x = coarsen_x ? coarsened_begin(level.global_begin_x,factor_x) :
                                             level.global_begin_x;
      int const coarse_begin_y = coarsen_y ? coarsened_begin(level.global_begin_y,factor_y) :
                                             level.global_begin_y;
      int const coarse_end_x = coarsen_x ? coarsened_begin(level.global_begin_x+level.nx,factor_x) :
                                           level.global_begin_x+level.nx;
      int const coarse_end_y = coarsen_y ? coarsened_begin(level.global_begin_y+level.ny,factor_y) :
                                           level.global_begin_y+level.ny;
      int const local_nx = coarse_end_x-coarse_begin_x;
      int const local_ny = coarse_end_y-coarse_begin_y;
      int const local_nz = coarsened_extent(level.nz,factor_z);
      long long const local_cells = static_cast<long long>(local_nx)*local_ny*local_nz;
      long long cells_per_rank = 0;
      MPI_Allreduce(&local_cells,&cells_per_rank,1,MPI_LONG_LONG,MPI_MIN,level.comm);
      int group_x = 1;
      int group_y = 1;
      cells_per_rank *= group_x*group_y;
      if (cells_per_rank < options.min_cells_per_rank) {
        bool const can_group_x = group_x == 1 && level.nproc_x > 1;
        bool const can_group_y = group_y == 1 && level.nproc_y > 1;
        if (can_group_x && (!can_group_y || level.nproc_x >= level.nproc_y)) {
          group_x = 2;
          cells_per_rank *= 2;
        } else if (can_group_y) {
          group_y = 2;
          cells_per_rank *= 2;
        }
        if (cells_per_rank < options.min_cells_per_rank) {
          if (group_x == 1 && level.nproc_x > 1) group_x = 2;
          if (group_y == 1 && level.nproc_y > 1) group_y = 2;
        }
      }
      if (!coarsen_x && !coarsen_y && !coarsen_z && level.nranks > 1) {
        if (level.nproc_x > 1) group_x = 2;
        if (level.nproc_y > 1) group_y = 2;
      }
      if (!coarsen_x && !coarsen_y && !coarsen_z && group_x == 1 && group_y == 1) {
        endrun("ERROR: geometric multigrid hierarchy construction made no progress");
      }

      auto transition = std::make_unique<Transition>();
      transition->coarsen_x = coarsen_x;
      transition->coarsen_y = coarsen_y;
      transition->coarsen_z = coarsen_z;
      transition->periodic_x = level.periodic_x;
      transition->periodic_y = level.periodic_y;
      transition->periodic_z = level.periodic_z;
      transition->leader = level.px%group_x == 0 && level.py%group_y == 0;
      int const leader_px = (level.px/group_x)*group_x;
      int const leader_py = (level.py/group_y)*group_y;
      transition->leader_rank = leader_py*level.nproc_x+leader_px;
      transition->local_nx = local_nx;
      transition->local_ny = local_ny;
      transition->local_nz = local_nz;
      transition->coarse_begin_x = coarse_begin_x;
      transition->coarse_begin_y = coarse_begin_y;
      transition->factor_x = factor_x;
      transition->factor_y = factor_y;
      transition->factor_z = factor_z;
      transition->restrict_x = Device3d("geometric_multigrid_restrict_x",level.nz,level.ny,local_nx);
      transition->restrict_y = Device3d("geometric_multigrid_restrict_y",level.nz,local_ny,local_nx);
      int restrict_halo_width = 0;
      int prolong_halo_width = 0;
      if (coarsen_x) {
        initialize_transfer(level.dx_global_host,factor_x,level.periodic_x,level.global_begin_x,level.nx,
                            coarse_begin_x,local_nx,"geometric_multigrid_x",transition->map_x,
                            restrict_halo_width,prolong_halo_width);
        if (restrict_halo_width > 2*factor_x || prolong_halo_width > 1) {
          endrun("ERROR: geometric multigrid x transfer exceeded its MPI halo");
        }
        allocate_transfer_halo(transition->restrict_x_halo,level.nproc_x > 1 ? 2*factor_x : 0,
                               level.nz,level.ny,level.nx,true,"geometric_multigrid_restrict_x_halo");
        allocate_transfer_halo(transition->prolong_x_halo,level.nproc_x > 1 ? 1 : 0,
                               level.nz,level.ny,local_nx,true,"geometric_multigrid_prolong_x_halo");
      }
      if (coarsen_y) {
        initialize_transfer(level.dy_global_host,factor_y,level.periodic_y,level.global_begin_y,level.ny,
                            coarse_begin_y,local_ny,"geometric_multigrid_y",transition->map_y,
                            restrict_halo_width,prolong_halo_width);
        if (restrict_halo_width > 2*factor_y || prolong_halo_width > 1) {
          endrun("ERROR: geometric multigrid y transfer exceeded its MPI halo");
        }
        allocate_transfer_halo(transition->restrict_y_halo,level.nproc_y > 1 ? 2*factor_y : 0,
                               level.nz,level.ny,local_nx,false,"geometric_multigrid_restrict_y_halo");
        allocate_transfer_halo(transition->prolong_y_halo,level.nproc_y > 1 ? 1 : 0,
                               level.nz,local_ny,local_nx,false,"geometric_multigrid_prolong_y_halo");
      }
      if (coarsen_z) {
        transition->local_coarse = Device3d("geometric_multigrid_local_coarse",local_nz,local_ny,local_nx);
        transition->prolong_z = Device3d("geometric_multigrid_prolong_z",level.nz,local_ny,local_nx);
        initialize_transfer(level.dz_host,factor_z,level.periodic_z,0,level.nz,0,local_nz,
                            "geometric_multigrid_z",transition->map_z,
                            restrict_halo_width,prolong_halo_width);
      } else {
        transition->local_coarse = transition->restrict_y;
        transition->prolong_z = transition->local_coarse;
      }
      transition->prolong_y = Device3d("geometric_multigrid_prolong_y",level.nz,level.ny,local_nx);
      transition->local_host = Host3d("geometric_multigrid_local_host",local_nz,local_ny,local_nx);

      std::vector<int> dimensions(2*level.nranks);
      int local_dimensions[2] = {local_nx,local_ny};
      MPI_Allgather(local_dimensions,2,MPI_INT,dimensions.data(),2,MPI_INT,level.comm);
      int coarse_nx = 0;
      int coarse_ny = 0;
      int const coarse_nx_global = coarsened_extent(level.nx_global,factor_x);
      int const coarse_ny_global = coarsened_extent(level.ny_global,factor_y);
      int const group_end_x = std::min(leader_px+group_x,level.nproc_x);
      int const group_end_y = std::min(leader_py+group_y,level.nproc_y);
      for (int px = leader_px; px < group_end_x; px++) coarse_nx += dimensions[2*(leader_py*level.nproc_x+px)];
      for (int py = leader_py; py < group_end_y; py++) coarse_ny += dimensions[2*(py*level.nproc_x+leader_px)+1];
      if (transition->leader) {
        int oy = 0;
        for (int py = leader_py; py < group_end_y; py++) {
          int ox = 0;
          int const block_ny = dimensions[2*(py*level.nproc_x+leader_px)+1];
          for (int px = leader_px; px < group_end_x; px++) {
            int const rank = py*level.nproc_x+px;
            int const block_nx = dimensions[2*rank];
            Block block;
            block.rank = rank;
            block.nx = block_nx;
            block.ny = block_ny;
            block.ox = ox;
            block.oy = oy;
            block.device = Device3d("geometric_multigrid_group_block",local_nz,block_ny,block_nx);
            block.host = Host3d("geometric_multigrid_group_block_host",local_nz,block_ny,block_nx);
            transition->blocks.push_back(block);
            ox += block_nx;
          }
          oy += block_ny;
        }
      }
      level.transition = std::move(transition);

      MPI_Comm next_comm = MPI_COMM_NULL;
      MPI_Comm_split(level.comm,level.transition->leader ? 0 : MPI_UNDEFINED,level.rank,&next_comm);
      if (!level.transition->leader) {
        active = false;
        break;
      }
      auto coarse = std::make_unique<Level>();
      coarse->nx = coarse_nx;
      coarse->ny = coarse_ny;
      coarse->nz = local_nz;
      coarse->nx_global = coarse_nx_global;
      coarse->ny_global = coarse_ny_global;
      coarse->nproc_x = (level.nproc_x+group_x-1)/group_x;
      coarse->nproc_y = (level.nproc_y+group_y-1)/group_y;
      coarse->px = level.px/group_x;
      coarse->py = level.py/group_y;
      coarse->comm = next_comm;
      coarse->owns_comm = true;
      MPI_Comm_rank(next_comm,&coarse->rank);
      MPI_Comm_size(next_comm,&coarse->nranks);
      coarse->global_begin_x = coarse_begin_x;
      coarse->global_begin_y = coarse_begin_y;
      coarse->periodic_x = level.periodic_x;
      coarse->periodic_y = level.periodic_y;
      coarse->periodic_z = level.periodic_z;
      coarse->dx_global_host = coarsen_widths(level.dx_global_host,factor_x);
      coarse->dy_global_host = coarsen_widths(level.dy_global_host,factor_y);
      coarse->dx_host.assign(coarse->dx_global_host.begin()+coarse->global_begin_x,
                             coarse->dx_global_host.begin()+coarse->global_begin_x+coarse->nx);
      coarse->dy_host.assign(coarse->dy_global_host.begin()+coarse->global_begin_y,
                             coarse->dy_global_host.begin()+coarse->global_begin_y+coarse->ny);
      coarse->dz_host = coarsen_widths(level.dz_host,factor_z);
      allocate_level(*coarse);
      levels_.push_back(std::move(coarse));
      level_index++;
    }

    int metadata[6] = {0,0,0,0,0,0};
    if (coupler.get_myrank() == 0) {
      Level const &coarse = *levels_.back();
      metadata[0] = levels_.size();
      metadata[1] = coarse.nx_global*coarse.ny_global*coarse.nz;
      metadata[2] = coarse.nranks;
      metadata[3] = coarse.nx_global;
      metadata[4] = coarse.ny_global;
      metadata[5] = coarse.nz;
    }
    MPI_Bcast(metadata,6,MPI_INT,0,root_comm_);
    int local_uses_mpi_transfers = uses_mpi_transfers_ ? 1 : 0;
    int global_uses_mpi_transfers = 0;
    MPI_Allreduce(&local_uses_mpi_transfers,&global_uses_mpi_transfers,1,MPI_INT,MPI_MAX,root_comm_);
    if (metadata[2] != 1) {
      endrun("ERROR: geometric multigrid failed to aggregate its coarse grid onto one task");
    }
    std::string const metadata_prefix = "dycore_anelastic_geometric_multigrid";
    coupler.set_option<int>(metadata_prefix+"_levels",metadata[0]);
    coupler.set_option<int>(metadata_prefix+"_coarse_cells",metadata[1]);
    coupler.set_option<int>(metadata_prefix+"_coarse_ranks",metadata[2]);
    coupler.set_option<int>(metadata_prefix+"_coarse_nx",metadata[3]);
    coupler.set_option<int>(metadata_prefix+"_coarse_ny",metadata[4]);
    coupler.set_option<int>(metadata_prefix+"_coarse_nz",metadata[5]);
    coupler.set_option<int>(metadata_prefix+"_coarse_columns",metadata[3]*metadata[4]);
    coupler.set_option<int>(metadata_prefix+"_coarsening_factor",coarsening_factor_);
    coupler.set_option<bool>(metadata_prefix+"_uses_mpi_transfers",global_uses_mpi_transfers != 0);
    #ifdef PORTURB_GPU_AWARE_MPI
      coupler.set_option<bool>(metadata_prefix+"_gpu_aware_mpi",true);
    #else
      coupler.set_option<bool>(metadata_prefix+"_gpu_aware_mpi",false);
    #endif
    coupler.set_option<std::string>(metadata_prefix+"_interpolation","PhysicalCoordinateQuadratic");
    coupler.set_option<std::string>(metadata_prefix+"_smoother","Jacobi");
    coupler.set_option<std::string>(metadata_prefix+"_coarse_smoother","Jacobi");
    initialized_ = true;
  }


  void apply(yakl::Array<Scalar *> const &r, yakl::Array<Scalar *> const &z,
             Scalar screening_inverse_length_squared, Scalar dt) const {
    if constexpr (yakl::yakl_auto_profile) yakl::timer_start("geometric_multigrid_apply");
    if (!initialized_) endrun("ERROR: applying an uninitialized geometric multigrid preconditioner");
    if (r.size() != fine_size_ || z.size() != fine_size_) {
      endrun("ERROR: geometric multigrid input/output size does not match the initialized fine grid");
    }
    Level &fine = *levels_.front();
    auto fine_b = fine.b;
    auto fine_x = fine.x;
    Scalar const inverse_dt = Scalar(1)/dt;
    int const nx = fine.nx;
    int const ny = fine.ny;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,fine.nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      int const cell = i+nx*(j+ny*k);
      fine_b(k,j,i) = r(cell)*inverse_dt;
      fine_x(k,j,i) = 0;
      z(cell) = 0;
    });
    for (int cycle = 0; cycle < vcycles_; cycle++) vcycle(0,screening_inverse_length_squared);
    fine_x = fine.x;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,fine.nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      int const cell = i+nx*(j+ny*k);
      z(cell) = fine_x(k,j,i);
    });
    if constexpr (yakl::yakl_auto_profile) yakl::timer_stop("geometric_multigrid_apply");
  }
};


} // namespace modules
