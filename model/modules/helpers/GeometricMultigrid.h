#pragma once

#include "main_header.h"
#include "coupler.h"
#include <array>
#include <memory>
#include <vector>

namespace modules {


// Structured, matrix-free geometric multigrid for a pure-fluid pressure projection. The hierarchy uses quadratic
// cell-centered interpolation and its scaled transpose, so a symmetric V-cycle remains suitable for CG. Horizontal
// rank agglomeration keeps coarse grids large enough for GPUs and removes MPI tasks as the grid shrinks.
template <class Scalar>
class GeometricMultigrid {
public:
  using Device3d = yakl::Array<Scalar ***>;
  using Device2d = yakl::Array<Scalar **>;
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
    bool periodic_x = false;
    bool periodic_y = false;
    bool periodic_z = false;
    bool owns_comm = false;
    MPI_Comm comm = MPI_COMM_NULL;
    std::vector<Scalar> dx_host;
    std::vector<Scalar> dy_host;
    std::vector<Scalar> dz_host;
    yakl::Array<Scalar *> x_minus;
    yakl::Array<Scalar *> x_plus;
    yakl::Array<Scalar *> y_minus;
    yakl::Array<Scalar *> y_plus;
    yakl::Array<Scalar *> z_minus;
    yakl::Array<Scalar *> z_plus;
    Device3d x;
    Device3d b;
    Device3d residual;
    Device3d Ax;
    Device3d x_next;
    Device3d line_cprime;
    Device3d line_rhs;
    Device3d line_aux;
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
  bool vertical_line_smoother_ = false;
  bool horizontal_only_ = false;
  bool require_single_coarse_rank_ = false;
  int coarse_nx_ = 0;
  int coarse_ny_ = 0;
  std::string metadata_prefix_ = "dycore_anelastic_geometric_multigrid";
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


  KOKKOS_INLINE_FUNCTION static Scalar quadratic_weight(int fine_parity, int offset) {
    if (fine_parity == 0) {
      return offset == -1 ? Scalar(5)/Scalar(32) :
             offset ==  0 ? Scalar(15)/Scalar(16) : -Scalar(3)/Scalar(32);
    }
    return offset == -1 ? -Scalar(3)/Scalar(32) :
           offset ==  0 ?  Scalar(15)/Scalar(16) : Scalar(5)/Scalar(32);
  }


  KOKKOS_INLINE_FUNCTION static Scalar interpolation_weight(int fine, int coarse, int coarse_extent,
                                                             bool periodic) {
    int const parent = fine/2;
    int const parity = fine%2;
    Scalar weight = 0;
    for (int offset = -1; offset <= 1; offset++) {
      if (map_index(parent+offset,coarse_extent,periodic) == coarse) {
        weight += quadratic_weight(parity,offset);
      }
    }
    return weight;
  }


  static std::vector<Scalar> coarsen_dz(std::vector<Scalar> const &fine, bool coarsen) {
    if (!coarsen) return fine;
    std::vector<Scalar> coarse((fine.size()+1)/2,0);
    for (int k = 0; k < coarse.size(); k++) {
      coarse[k] = fine[2*k];
      if (2*k+1 < fine.size()) coarse[k] += fine[2*k+1];
    }
    return coarse;
  }


  static std::vector<Scalar> coarsen_widths(std::vector<Scalar> const &fine, bool coarsen) {
    return coarsen_dz(fine,coarsen);
  }


  void allocate_level(Level &level) const {
    level.x        = Device3d("geometric_multigrid_x"       ,level.nz,level.ny,level.nx);
    level.b        = Device3d("geometric_multigrid_b"       ,level.nz,level.ny,level.nx);
    level.residual = Device3d("geometric_multigrid_residual",level.nz,level.ny,level.nx);
    level.Ax       = Device3d("geometric_multigrid_Ax"      ,level.nz,level.ny,level.nx);
    if (vertical_line_smoother_) {
      level.x_next     = Device3d("tensor_line_multigrid_x_next"    ,level.nz,level.ny,level.nx);
      level.line_cprime = Device3d("tensor_line_multigrid_cprime"   ,level.nz,level.ny,level.nx);
      level.line_rhs    = Device3d("tensor_line_multigrid_line_rhs" ,level.nz,level.ny,level.nx);
      level.line_aux    = Device3d("tensor_line_multigrid_line_aux" ,level.nz,level.ny,level.nx);
      level.x_next = 0;
    }
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
    for (int i = 0; i < level.nx; i++) {
      x_minus_host(i) = 0;
      x_plus_host(i) = 0;
      if (i > 0 || west != MPI_PROC_NULL) {
        Scalar const adjacent = i > 0 ? level.dx_host[i-1] : boundary_widths[4*west+1];
        Scalar const distance = Scalar(0.5)*(level.dx_host[i]+adjacent);
        x_minus_host(i) = Scalar(1)/(distance*distance);
      }
      if (i+1 < level.nx || east != MPI_PROC_NULL) {
        Scalar const adjacent = i+1 < level.nx ? level.dx_host[i+1] : boundary_widths[4*east];
        Scalar const distance = Scalar(0.5)*(level.dx_host[i]+adjacent);
        x_plus_host(i) = Scalar(1)/(distance*distance);
      }
    }
    for (int j = 0; j < level.ny; j++) {
      y_minus_host(j) = 0;
      y_plus_host(j) = 0;
      if (j > 0 || south != MPI_PROC_NULL) {
        Scalar const adjacent = j > 0 ? level.dy_host[j-1] : boundary_widths[4*south+3];
        Scalar const distance = Scalar(0.5)*(level.dy_host[j]+adjacent);
        y_minus_host(j) = Scalar(1)/(distance*distance);
      }
      if (j+1 < level.ny || north != MPI_PROC_NULL) {
        Scalar const adjacent = j+1 < level.ny ? level.dy_host[j+1] : boundary_widths[4*north+2];
        Scalar const distance = Scalar(0.5)*(level.dy_host[j]+adjacent);
        y_plus_host(j) = Scalar(1)/(distance*distance);
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
        z_minus_host(k) = Scalar(1)/(distance*distance);
      }
      if (k+1 < level.nz || level.periodic_z) {
        int const kp = k+1 < level.nz ? k+1 : 0;
        Scalar const distance = Scalar(0.5)*(level.dz_host[k]+level.dz_host[kp]);
        z_plus_host(k) = Scalar(1)/(distance*distance);
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

    int const west = level.nproc_x > 1 ? neighbor(level,-1,0) : MPI_PROC_NULL;
    int const east = level.nproc_x > 1 ? neighbor(level, 1,0) : MPI_PROC_NULL;
    int const south = level.nproc_y > 1 ? neighbor(level,0,-1) : MPI_PROC_NULL;
    int const north = level.nproc_y > 1 ? neighbor(level,0, 1) : MPI_PROC_NULL;
    int const tag = 100+4*level_index;
    Exchange exchange;
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
    return exchange;
  }


  static void finish_exchange(Level &level, Exchange &exchange) {
    if (exchange.count > 0) MPI_Waitall(exchange.count,exchange.requests.data(),MPI_STATUSES_IGNORE);
    #ifndef PORTURB_GPU_AWARE_MPI
      if (level.nproc_x > 1) {
        level.recv_west_host.deep_copy_to(level.recv_west);
        level.recv_east_host.deep_copy_to(level.recv_east);
      }
      if (level.nproc_y > 1) {
        level.recv_south_host.deep_copy_to(level.recv_south);
        level.recv_north_host.deep_copy_to(level.recv_north);
      }
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


  void matvec(Level &level, Device3d const &x, Device3d const &Ax, Scalar shift, int level_index) const {
    level.x = x;
    Exchange exchange = begin_exchange(level,level_index);
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
    if (level.nx > 2 && level.ny > 2) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(level.nz,level.ny-2,level.nx-2),
                         KOKKOS_LAMBDA (int k, int jj, int ii) {
        int const j = jj+1;
        int const i = ii+1;
        Ax(k,j,i) = apply_cell(x,recv_west,recv_east,recv_south,recv_north,z_minus,z_plus,
                               x_minus,x_plus,y_minus,y_plus,nx,ny,nz,nproc_x,nproc_y,px,py,
                               periodic_x,periodic_y,k,j,i,shift);
      });
    }
    finish_exchange(level,exchange);
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(level.nz,level.ny,level.nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      if (i > 0 && i+1 < nx && j > 0 && j+1 < ny) return;
      Ax(k,j,i) = apply_cell(x,recv_west,recv_east,recv_south,recv_north,z_minus,z_plus,
                             x_minus,x_plus,y_minus,y_plus,nx,ny,nz,nproc_x,nproc_y,px,py,
                             periodic_x,periodic_y,k,j,i,shift);
    });
  }


  void smooth(Level &level, int iterations, Scalar shift, int level_index) const {
    auto x = level.x;
    auto b = level.b;
    auto Ax = level.Ax;
    auto z_minus = level.z_minus;
    auto z_plus = level.z_plus;
    auto x_minus = level.x_minus;
    auto x_plus = level.x_plus;
    auto y_minus = level.y_minus;
    auto y_plus = level.y_plus;
    Scalar const weight = jacobi_weight_;
    for (int iteration = 0; iteration < iterations; iteration++) {
      matvec(level,x,Ax,shift,level_index);
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(level.nz,level.ny,level.nx),
                         KOKKOS_LAMBDA (int k, int j, int i) {
        Scalar diagonal = shift+z_minus(k)+z_plus(k);
        diagonal += x_minus(i)+x_plus(i)+y_minus(j)+y_plus(j);
        if (diagonal > std::numeric_limits<Scalar>::min()) {
          x(k,j,i) += weight*(b(k,j,i)-Ax(k,j,i))/diagonal;
        }
      });
    }
  }


  KOKKOS_INLINE_FUNCTION static Scalar horizontal_neighbor_sum(
      Device3d const &x, Device2d const &recv_west, Device2d const &recv_east,
      Device2d const &recv_south, Device2d const &recv_north,
      yakl::Array<Scalar *> const &x_minus, yakl::Array<Scalar *> const &x_plus,
      yakl::Array<Scalar *> const &y_minus, yakl::Array<Scalar *> const &y_plus,
      int nx, int ny, int nproc_x, int nproc_y, int px, int py, bool periodic_x, bool periodic_y,
      int k, int j, int i) {
    Scalar value = 0;
    if (i > 0) {
      value += x_minus(i)*x(k,j,i-1);
    } else if (nproc_x > 1 && (periodic_x || px > 0)) {
      value += x_minus(i)*recv_west(k,j);
    } else if (periodic_x) {
      value += x_minus(i)*x(k,j,nx-1);
    }
    if (i+1 < nx) {
      value += x_plus(i)*x(k,j,i+1);
    } else if (nproc_x > 1 && (periodic_x || px+1 < nproc_x)) {
      value += x_plus(i)*recv_east(k,j);
    } else if (periodic_x) {
      value += x_plus(i)*x(k,j,0);
    }
    if (j > 0) {
      value += y_minus(j)*x(k,j-1,i);
    } else if (nproc_y > 1 && (periodic_y || py > 0)) {
      value += y_minus(j)*recv_south(k,i);
    } else if (periodic_y) {
      value += y_minus(j)*x(k,ny-1,i);
    }
    if (j+1 < ny) {
      value += y_plus(j)*x(k,j+1,i);
    } else if (nproc_y > 1 && (periodic_y || py+1 < nproc_y)) {
      value += y_plus(j)*recv_north(k,i);
    } else if (periodic_y) {
      value += y_plus(j)*x(k,0,i);
    }
    return value;
  }


  // One horizontal block-Jacobi iteration. Each horizontal point owns a complete vertical column, so a single
  // portable kernel forms the lagged horizontal right-hand side and applies a Thomas (or cyclic Thomas) solve.
  // On the one-rank terminal level begin_exchange launches no packing kernels, making each iteration one launch.
  void smooth_vertical_lines(Level &level, int iterations, Scalar shift, int level_index,
                             bool zero_initial_guess = false) const {
    int const nx = level.nx;
    int const ny = level.ny;
    int const nz = level.nz;
    int const nproc_x = level.nproc_x;
    int const nproc_y = level.nproc_y;
    int const px = level.px;
    int const py = level.py;
    bool const periodic_x = level.periodic_x;
    bool const periodic_y = level.periodic_y;
    bool const periodic_z = level.periodic_z;
    Scalar const weight = jacobi_weight_;
    auto b = level.b;
    auto recv_west = level.recv_west;
    auto recv_east = level.recv_east;
    auto recv_south = level.recv_south;
    auto recv_north = level.recv_north;
    auto x_minus = level.x_minus;
    auto x_plus = level.x_plus;
    auto y_minus = level.y_minus;
    auto y_plus = level.y_plus;
    auto z_minus = level.z_minus;
    auto z_plus = level.z_plus;
    for (int iteration = 0; iteration < iterations; iteration++) {
      bool const zero_guess = zero_initial_guess && iteration == 0;
      Exchange exchange = begin_exchange(level,level_index);
      finish_exchange(level,exchange);
      auto x = level.x;
      auto x_next = level.x_next;
      auto cprime = level.line_cprime;
      auto line_rhs = level.line_rhs;
      auto line_aux = level.line_aux;
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<2>(ny,nx),KOKKOS_LAMBDA (int j, int i) {
        Scalar const horizontal_diagonal = x_minus(i)+x_plus(i)+y_minus(j)+y_plus(j);
        if (nz == 1) {
          Scalar const diagonal = shift+horizontal_diagonal;
          Scalar const rhs = b(0,j,i)+(zero_guess ? Scalar(0) : horizontal_neighbor_sum(
              x,recv_west,recv_east,recv_south,recv_north,x_minus,x_plus,y_minus,y_plus,
              nx,ny,nproc_x,nproc_y,px,py,periodic_x,periodic_y,0,j,i));
          Scalar const solved = diagonal > std::numeric_limits<Scalar>::min() ? rhs/diagonal : Scalar(0);
          x_next(0,j,i) = (Scalar(1)-weight)*(zero_guess ? Scalar(0) : x(0,j,i))+weight*solved;
          return;
        }

        if (periodic_z && nz == 2) {
          Scalar const d0 = shift+horizontal_diagonal+z_minus(0)+z_plus(0);
          Scalar const d1 = shift+horizontal_diagonal+z_minus(1)+z_plus(1);
          Scalar const o0 = -(z_minus(0)+z_plus(0));
          Scalar const o1 = -(z_minus(1)+z_plus(1));
          Scalar const r0 = b(0,j,i)+(zero_guess ? Scalar(0) : horizontal_neighbor_sum(
              x,recv_west,recv_east,recv_south,recv_north,x_minus,x_plus,y_minus,y_plus,
              nx,ny,nproc_x,nproc_y,px,py,periodic_x,periodic_y,0,j,i));
          Scalar const r1 = b(1,j,i)+(zero_guess ? Scalar(0) : horizontal_neighbor_sum(
              x,recv_west,recv_east,recv_south,recv_north,x_minus,x_plus,y_minus,y_plus,
              nx,ny,nproc_x,nproc_y,px,py,periodic_x,periodic_y,1,j,i));
          Scalar const determinant = d0*d1-o0*o1;
          Scalar const inverse_determinant = std::abs(determinant) > std::numeric_limits<Scalar>::min() ?
                                             Scalar(1)/determinant : Scalar(0);
          Scalar const solved0 = (d1*r0-o0*r1)*inverse_determinant;
          Scalar const solved1 = (d0*r1-o1*r0)*inverse_determinant;
          x_next(0,j,i) = (Scalar(1)-weight)*(zero_guess ? Scalar(0) : x(0,j,i))+weight*solved0;
          x_next(1,j,i) = (Scalar(1)-weight)*(zero_guess ? Scalar(0) : x(1,j,i))+weight*solved1;
          return;
        }

        Scalar gamma = 0;
        Scalar alpha = 0;
        Scalar beta = 0;
        if (periodic_z) {
          Scalar const diagonal0 = shift+horizontal_diagonal+z_minus(0)+z_plus(0);
          gamma = -diagonal0;
          alpha = -z_minus(0);
          beta = -z_plus(nz-1);
        }
        for (int k = 0; k < nz; k++) {
          Scalar diagonal = shift+horizontal_diagonal+z_minus(k)+z_plus(k);
          Scalar const lower = k > 0 ? -z_minus(k) : Scalar(0);
          Scalar const upper = k+1 < nz ? -z_plus(k) : Scalar(0);
          Scalar rhs = b(k,j,i)+(zero_guess ? Scalar(0) : horizontal_neighbor_sum(
              x,recv_west,recv_east,recv_south,recv_north,x_minus,x_plus,y_minus,y_plus,
              nx,ny,nproc_x,nproc_y,px,py,periodic_x,periodic_y,k,j,i));
          Scalar auxiliary_rhs = 0;
          if (periodic_z) {
            if (k == 0) {
              diagonal -= gamma;
              auxiliary_rhs = gamma;
            } else if (k+1 == nz) {
              diagonal -= alpha*beta/gamma;
              auxiliary_rhs = alpha;
            }
          }
          Scalar const denominator = k == 0 ? diagonal : diagonal-lower*cprime(k-1,j,i);
          Scalar const inverse_denominator = Scalar(1)/denominator;
          cprime(k,j,i) = upper*inverse_denominator;
          line_rhs(k,j,i) = (rhs-(k == 0 ? Scalar(0) : lower*line_rhs(k-1,j,i)))*inverse_denominator;
          line_aux(k,j,i) = (auxiliary_rhs-(k == 0 ? Scalar(0) : lower*line_aux(k-1,j,i)))*inverse_denominator;
        }
        for (int k = nz-2; k >= 0; k--) {
          line_rhs(k,j,i) -= cprime(k,j,i)*line_rhs(k+1,j,i);
          if (periodic_z) line_aux(k,j,i) -= cprime(k,j,i)*line_aux(k+1,j,i);
        }
        Scalar correction = 0;
        if (periodic_z) {
          correction = (line_rhs(0,j,i)+beta*line_rhs(nz-1,j,i)/gamma) /
                       (Scalar(1)+line_aux(0,j,i)+beta*line_aux(nz-1,j,i)/gamma);
        }
        for (int k = 0; k < nz; k++) {
          Scalar const solved = line_rhs(k,j,i)-correction*line_aux(k,j,i);
          x_next(k,j,i) = (Scalar(1)-weight)*(zero_guess ? Scalar(0) : x(k,j,i))+weight*solved;
        }
      });
      level.x = x_next;
      level.x_next = x;
    }
  }


  void apply_smoother(Level &level, int iterations, Scalar shift, int level_index,
                      bool zero_initial_guess = false) const {
    if (vertical_line_smoother_) {
      smooth_vertical_lines(level,iterations,shift,level_index,zero_initial_guess);
    } else {
      if (zero_initial_guess) level.x = 0;
      smooth(level,iterations,shift,level_index);
    }
  }


  static void restrict_dimension_x(Level const &fine, Transition &transition) {
    auto input = fine.residual;
    auto output = transition.restrict_x;
    if (!transition.coarsen_x) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,fine.nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) = input(k,j,i); });
      return;
    }
    int const nc = transition.local_nx;
    bool const periodic = transition.periodic_x;
    int const fine_nx = fine.nx;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,nc),
                       KOKKOS_LAMBDA (int k, int j, int q) {
      Scalar value = 0;
      for (int p_offset = -1; p_offset <= 1; p_offset++) {
        int const p = map_index(q+p_offset,nc,periodic);
        bool duplicate = false;
        for (int previous = -1; previous < p_offset; previous++) {
          if (map_index(q+previous,nc,periodic) == p) duplicate = true;
        }
        if (duplicate) continue;
        for (int parity = 0; parity < 2; parity++) {
          int const i = 2*p+parity;
          if (i < fine_nx) value += interpolation_weight(i,q,nc,periodic)*input(k,j,i);
        }
      }
      output(k,j,q) = Scalar(0.5)*value;
    });
  }


  static void restrict_dimension_y(Level const &fine, Transition &transition) {
    auto input = transition.restrict_x;
    auto output = transition.restrict_y;
    int const nx = transition.local_nx;
    if (!transition.coarsen_y) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) = input(k,j,i); });
      return;
    }
    int const nc = transition.local_ny;
    bool const periodic = transition.periodic_y;
    int const fine_ny = fine.ny;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,nc,nx),
                       KOKKOS_LAMBDA (int k, int q, int i) {
      Scalar value = 0;
      for (int p_offset = -1; p_offset <= 1; p_offset++) {
        int const p = map_index(q+p_offset,nc,periodic);
        bool duplicate = false;
        for (int previous = -1; previous < p_offset; previous++) {
          if (map_index(q+previous,nc,periodic) == p) duplicate = true;
        }
        if (duplicate) continue;
        for (int parity = 0; parity < 2; parity++) {
          int const j = 2*p+parity;
          if (j < fine_ny) value += interpolation_weight(j,q,nc,periodic)*input(k,j,i);
        }
      }
      output(k,q,i) = Scalar(0.5)*value;
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
    int const nc = transition.local_nz;
    bool const periodic = transition.periodic_z;
    int const fine_nz = fine.nz;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(nc,ny,nx),
                       KOKKOS_LAMBDA (int q, int j, int i) {
      Scalar value = 0;
      for (int p_offset = -1; p_offset <= 1; p_offset++) {
        int const p = map_index(q+p_offset,nc,periodic);
        bool duplicate = false;
        for (int previous = -1; previous < p_offset; previous++) {
          if (map_index(q+previous,nc,periodic) == p) duplicate = true;
        }
        if (duplicate) continue;
        for (int parity = 0; parity < 2; parity++) {
          int const k = 2*p+parity;
          if (k < fine_nz) value += interpolation_weight(k,q,nc,periodic)*input(k,j,i);
        }
      }
      output(q,j,i) = Scalar(0.5)*value;
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
    int const nc = transition.local_nz;
    bool const periodic = transition.periodic_z;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,ny,nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      int const parent = k/2;
      Scalar value = 0;
      for (int offset = -1; offset <= 1; offset++) {
        value += quadratic_weight(k%2,offset)*input(map_index(parent+offset,nc,periodic),j,i);
      }
      output(k,j,i) = value;
    });
  }


  static void prolong_dimension_y(Level const &fine, Transition &transition) {
    auto input = transition.prolong_z;
    auto output = transition.prolong_y;
    int const nx = transition.local_nx;
    if (!transition.coarsen_y) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) = input(k,j,i); });
      return;
    }
    int const nc = transition.local_ny;
    bool const periodic = transition.periodic_y;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      int const parent = j/2;
      Scalar value = 0;
      for (int offset = -1; offset <= 1; offset++) {
        value += quadratic_weight(j%2,offset)*input(k,map_index(parent+offset,nc,periodic),i);
      }
      output(k,j,i) = value;
    });
  }


  static void prolong_dimension_x(Level &fine, Transition &transition) {
    auto input = transition.prolong_y;
    auto output = fine.x;
    if (!transition.coarsen_x) {
      yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,fine.nx),
                         KOKKOS_LAMBDA (int k, int j, int i) { output(k,j,i) += input(k,j,i); });
      return;
    }
    int const nc = transition.local_nx;
    bool const periodic = transition.periodic_x;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(fine.nz,fine.ny,fine.nx),
                       KOKKOS_LAMBDA (int k, int j, int i) {
      int const parent = i/2;
      Scalar value = 0;
      for (int offset = -1; offset <= 1; offset++) {
        value += quadratic_weight(i%2,offset)*input(k,j,map_index(parent+offset,nc,periodic));
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
    Transition &transition = *fine.transition;
    int const tag = 1000+level_index;
    #ifdef PORTURB_GPU_AWARE_MPI
      Kokkos::fence();
      if (!transition.leader) {
        MPI_Send(transition.local_coarse.data(),transition.local_coarse.size(),mpi_scalar_type(),
                 transition.leader_rank,tag,fine.comm);
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
      if (!requests.empty()) MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE);
      for (auto &block : transition.blocks) {
        if (block.rank != fine.rank) copy_block_to_level(block.device,coarse->b,block.ox,block.oy);
      }
    #else
      transition.local_coarse.deep_copy_to(transition.local_host);
      if (!transition.leader) {
        MPI_Send(transition.local_host.data(),transition.local_host.size(),mpi_scalar_type(),
                 transition.leader_rank,tag,fine.comm);
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
      if (!requests.empty()) MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE);
      for (auto &block : transition.blocks) {
        if (block.rank != fine.rank) {
          block.host.deep_copy_to(block.device);
          copy_block_to_level(block.device,coarse->b,block.ox,block.oy);
        }
      }
    #endif
  }


  void scatter_correction(Level &fine, Level *coarse, int level_index) const {
    Transition &transition = *fine.transition;
    int const tag = 2000+level_index;
    #ifdef PORTURB_GPU_AWARE_MPI
      if (!transition.leader) {
        MPI_Recv(transition.local_coarse.data(),transition.local_coarse.size(),mpi_scalar_type(),
                 transition.leader_rank,tag,fine.comm,MPI_STATUS_IGNORE);
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
      if (!requests.empty()) MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE);
    #else
      if (!transition.leader) {
        MPI_Recv(transition.local_host.data(),transition.local_host.size(),mpi_scalar_type(),
                 transition.leader_rank,tag,fine.comm,MPI_STATUS_IGNORE);
        transition.local_host.deep_copy_to(transition.local_coarse);
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
      if (!requests.empty()) MPI_Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE);
    #endif
  }


  void vcycle(int level_index, Scalar shift, bool zero_initial_guess = false) const {
    Level &level = *levels_[level_index];
    if (!level.transition) {
      apply_smoother(level,coarse_smooth_,shift,level_index,zero_initial_guess);
      return;
    }
    apply_smoother(level,pre_smooth_,shift,level_index,zero_initial_guess);
    matvec(level,level.x,level.Ax,shift,level_index);
    auto residual = level.residual;
    auto b = level.b;
    auto Ax = level.Ax;
    yakl::parallel_for(YAKL_AUTO_LABEL(),yakl::SimpleBounds<3>(level.nz,level.ny,level.nx),
                       KOKKOS_LAMBDA (int k, int j, int i) { residual(k,j,i) = b(k,j,i)-Ax(k,j,i); });
    Transition &transition = *level.transition;
    restrict_dimension_x(level,transition);
    restrict_dimension_y(level,transition);
    if (transition.coarsen_z) restrict_dimension_z(level,transition);
    Level *coarse = transition.leader ? levels_[level_index+1].get() : nullptr;
    gather_restricted(level,coarse,level_index);
    if (transition.leader) {
      vcycle(level_index+1,shift,true);
    }
    scatter_correction(level,coarse,level_index);
    if (transition.coarsen_z) prolong_dimension_z(level,transition);
    prolong_dimension_y(level,transition);
    prolong_dimension_x(level,transition);
    apply_smoother(level,post_smooth_,shift,level_index);
  }


public:
  struct Options {
    int vcycles = 1;
    int pre_smooth = 2;
    int post_smooth = 2;
    int coarse_smooth = 24;
    int max_levels = 20;
    int coarse_cells = 32768;
    int min_cells_per_rank = 131072;
    Scalar jacobi_weight = Scalar(2)/Scalar(3);
    bool vertical_line_smoother = false;
    bool horizontal_only = false;
    bool require_single_coarse_rank = false;
    int coarse_nx = 0;
    int coarse_ny = 0;
    std::string metadata_prefix = "dycore_anelastic_geometric_multigrid";
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
        options.pre_smooth != options.post_smooth || options.coarse_smooth <= 0 || options.max_levels < 2 ||
        options.coarse_cells <= 0 || options.min_cells_per_rank <= 0 ||
        (options.horizontal_only && (options.coarse_nx <= 0 || options.coarse_ny <= 0)) ||
        options.metadata_prefix.empty() ||
        !(options.jacobi_weight > 0 && options.jacobi_weight < 1)) {
      endrun("ERROR: invalid geometric multigrid options; CG requires equal pre/post smoothing counts");
    }
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
    vertical_line_smoother_ = options.vertical_line_smoother;
    horizontal_only_ = options.horizontal_only;
    require_single_coarse_rank_ = options.require_single_coarse_rank;
    coarse_nx_ = options.coarse_nx;
    coarse_ny_ = options.coarse_ny;
    metadata_prefix_ = options.metadata_prefix;
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
    fine->periodic_x = bc_x1 == "periodic";
    fine->periodic_y = bc_y1 == "periodic";
    fine->periodic_z = bc_z1 == "periodic";
    fine->dx_host.assign(fine->nx,static_cast<Scalar>(coupler.get_dx()));
    fine->dy_host.assign(fine->ny,static_cast<Scalar>(coupler.get_dy()));
    fine->dz_host = dz;
    allocate_level(*fine);
    levels_.push_back(std::move(fine));

    bool active = true;
    for (int level_index = 0; level_index+1 < options.max_levels && active; level_index++) {
      Level &level = *levels_.back();
      long long const global_cells = static_cast<long long>(level.nx_global)*level.ny_global*level.nz;
      bool const horizontal_target = horizontal_only_ && level.nx_global <= coarse_nx_ &&
                                     level.ny_global <= coarse_ny_;
      if (level.nranks == 1 && (horizontal_target || (!horizontal_only_ && global_cells <= options.coarse_cells))) {
        break;
      }

      bool const coarsen_x = horizontal_only_ ?
          level.nx_global > coarse_nx_ && (level.nx_global+1)/2 >= coarse_nx_ : level.nx_global > 2;
      bool const coarsen_y = horizontal_only_ ?
          level.ny_global > coarse_ny_ && (level.ny_global+1)/2 >= coarse_ny_ : level.ny_global > 2;
      bool const coarsen_z = !horizontal_only_ && level.nz > 2;
      if (!coarsen_x && !coarsen_y && !coarsen_z && level.nranks == 1) break;

      int const local_nx = coarsen_x ? (level.nx+1)/2 : level.nx;
      int const local_ny = coarsen_y ? (level.ny+1)/2 : level.ny;
      int const local_nz = coarsen_z ? (level.nz+1)/2 : level.nz;
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
      if (horizontal_only_ && !coarsen_x && !coarsen_y && level.nranks > 1) {
        if (level.nproc_x > 1) group_x = 2;
        if (level.nproc_y > 1) group_y = 2;
      }

      auto transition = std::make_unique<Transition>();
      transition->coarsen_x = coarsen_x;
      transition->coarsen_y = coarsen_y;
      transition->coarsen_z = coarsen_z;
      transition->periodic_x = level.periodic_x && level.nproc_x == 1;
      transition->periodic_y = level.periodic_y && level.nproc_y == 1;
      transition->periodic_z = level.periodic_z;
      transition->leader = level.px%group_x == 0 && level.py%group_y == 0;
      int const leader_px = (level.px/group_x)*group_x;
      int const leader_py = (level.py/group_y)*group_y;
      transition->leader_rank = leader_py*level.nproc_x+leader_px;
      transition->local_nx = local_nx;
      transition->local_ny = local_ny;
      transition->local_nz = local_nz;
      transition->restrict_x = Device3d("geometric_multigrid_restrict_x",level.nz,level.ny,local_nx);
      transition->restrict_y = Device3d("geometric_multigrid_restrict_y",level.nz,local_ny,local_nx);
      if (coarsen_z) {
        transition->local_coarse = Device3d("geometric_multigrid_local_coarse",local_nz,local_ny,local_nx);
        transition->prolong_z = Device3d("geometric_multigrid_prolong_z",level.nz,local_ny,local_nx);
      } else {
        transition->local_coarse = transition->restrict_y;
        transition->prolong_z = transition->local_coarse;
      }
      transition->prolong_y = Device3d("geometric_multigrid_prolong_y",level.nz,level.ny,local_nx);
      transition->local_host = Host3d("geometric_multigrid_local_host",local_nz,local_ny,local_nx);

      std::vector<int> dimensions(2*level.nranks);
      int local_dimensions[2] = {local_nx,local_ny};
      MPI_Allgather(local_dimensions,2,MPI_INT,dimensions.data(),2,MPI_INT,level.comm);
      std::vector<Scalar> local_dx = coarsen_widths(level.dx_host,coarsen_x);
      std::vector<Scalar> local_dy = coarsen_widths(level.dy_host,coarsen_y);
      std::vector<int> x_counts(level.nranks);
      std::vector<int> y_counts(level.nranks);
      std::vector<int> x_displacements(level.nranks,0);
      std::vector<int> y_displacements(level.nranks,0);
      for (int rank = 0; rank < level.nranks; rank++) {
        x_counts[rank] = dimensions[2*rank];
        y_counts[rank] = dimensions[2*rank+1];
        if (rank > 0) {
          x_displacements[rank] = x_displacements[rank-1]+x_counts[rank-1];
          y_displacements[rank] = y_displacements[rank-1]+y_counts[rank-1];
        }
      }
      std::vector<Scalar> gathered_dx(x_displacements.back()+x_counts.back());
      std::vector<Scalar> gathered_dy(y_displacements.back()+y_counts.back());
      MPI_Allgatherv(local_dx.data(),local_dx.size(),mpi_scalar_type(),gathered_dx.data(),x_counts.data(),
                     x_displacements.data(),mpi_scalar_type(),level.comm);
      MPI_Allgatherv(local_dy.data(),local_dy.size(),mpi_scalar_type(),gathered_dy.data(),y_counts.data(),
                     y_displacements.data(),mpi_scalar_type(),level.comm);
      int coarse_nx = 0;
      int coarse_ny = 0;
      int coarse_nx_global = 0;
      int coarse_ny_global = 0;
      int const group_end_x = std::min(leader_px+group_x,level.nproc_x);
      int const group_end_y = std::min(leader_py+group_y,level.nproc_y);
      for (int px = leader_px; px < group_end_x; px++) coarse_nx += dimensions[2*(leader_py*level.nproc_x+px)];
      for (int py = leader_py; py < group_end_y; py++) coarse_ny += dimensions[2*(py*level.nproc_x+leader_px)+1];
      for (int px = 0; px < level.nproc_x; px++) coarse_nx_global += dimensions[2*px];
      for (int py = 0; py < level.nproc_y; py++) coarse_ny_global += dimensions[2*(py*level.nproc_x)+1];
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
      coarse->periodic_x = level.periodic_x;
      coarse->periodic_y = level.periodic_y;
      coarse->periodic_z = level.periodic_z;
      for (int px = leader_px; px < group_end_x; px++) {
        int const rank = leader_py*level.nproc_x+px;
        coarse->dx_host.insert(coarse->dx_host.end(),gathered_dx.begin()+x_displacements[rank],
                               gathered_dx.begin()+x_displacements[rank]+x_counts[rank]);
      }
      for (int py = leader_py; py < group_end_y; py++) {
        int const rank = py*level.nproc_x+leader_px;
        coarse->dy_host.insert(coarse->dy_host.end(),gathered_dy.begin()+y_displacements[rank],
                               gathered_dy.begin()+y_displacements[rank]+y_counts[rank]);
      }
      coarse->dz_host = coarsen_dz(level.dz_host,coarsen_z);
      allocate_level(*coarse);
      levels_.push_back(std::move(coarse));
    }

    int metadata[5] = {0,0,0,0,0};
    if (coupler.get_myrank() == 0) {
      Level const &coarse = *levels_.back();
      metadata[0] = levels_.size();
      metadata[1] = coarse.nx_global*coarse.ny_global*coarse.nz;
      metadata[2] = coarse.nranks;
      metadata[3] = coarse.nx_global;
      metadata[4] = coarse.ny_global;
    }
    MPI_Bcast(metadata,5,MPI_INT,0,root_comm_);
    if (require_single_coarse_rank_ && metadata[2] != 1) {
      endrun("ERROR: tensor-line multigrid exhausted max_levels before aggregating its coarse grid onto one task");
    }
    coupler.set_option<int>(metadata_prefix_+"_levels",metadata[0]);
    coupler.set_option<int>(metadata_prefix_+"_coarse_cells",metadata[1]);
    coupler.set_option<int>(metadata_prefix_+"_coarse_ranks",metadata[2]);
    coupler.set_option<int>(metadata_prefix_+"_coarse_nx",metadata[3]);
    coupler.set_option<int>(metadata_prefix_+"_coarse_ny",metadata[4]);
    coupler.set_option<int>(metadata_prefix_+"_coarse_columns",metadata[3]*metadata[4]);
    #ifdef PORTURB_GPU_AWARE_MPI
      coupler.set_option<bool>(metadata_prefix_+"_gpu_aware_mpi",true);
    #else
      coupler.set_option<bool>(metadata_prefix_+"_gpu_aware_mpi",false);
    #endif
    coupler.set_option<std::string>(metadata_prefix_+"_interpolation","Quadratic");
    coupler.set_option<std::string>(metadata_prefix_+"_smoother",
                                    vertical_line_smoother_ ? "HorizontalJacobiVerticalTridiagonal" : "Jacobi");
    initialized_ = true;
  }


  void apply(yakl::Array<Scalar *> const &r, yakl::Array<Scalar *> const &z,
             Scalar screening_inverse_length_squared, Scalar dt) const {
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
  }
};


} // namespace modules
