#include "TriMesh.h"

namespace modules {

std::ostream &operator<<(std::ostream& os, TriMesh::Vertex const &v ) {
        os << "[" << v.x << " , " << v.y << " , " << v.z << "]";
        return os;
      }

void TriMesh::load_file(std::string fname) {
      float constexpr pos_huge = std::numeric_limits<float>::max();    // Highest possible float
      float constexpr neg_huge = std::numeric_limits<float>::lowest(); // Lowest possible float
      float xl = pos_huge, yl = pos_huge, zl = pos_huge; // Keep track of lower bounds of domain
      float xh = neg_huge, yh = neg_huge, zh = neg_huge; // Keep track of upper bounds of domain
      std::ifstream file(fname);      // Read file as a stream
      std::string line;               // Line for getline to store into
      std::vector<Vertex> vertices;   // List of vertices from wavefront obj file
      std::vector<Face>   faces_vec;  // List of triangular faces from wavefront obj file
      // Loop through file lines
      while (std::getline(file, line)) {
        // if the line isn't empty
        if (line.size() > 0) {
          // Lines starting with the letter 'v' define vertices
          if (line[0] == 'v') {
            std::string lab;
            float x, y, z;
            std::stringstream(line) >> lab >> x >> y >> z;
            // Track domain extents while reading in vertices
            xl = std::min(xl,x);  yl = std::min(yl,y);  zl = std::min(zl,z);
            xh = std::max(xh,x);  yh = std::max(yh,y);  zh = std::max(zh,z);
            vertices.push_back({x,y,z});
          }
          // Lines starting with the letter 'v' define faces using vertex indices using one-based indexing
          if (line[0] == 'f') {
            std::string lab, stri, strj, strk;
            int i, j, k;
            std::stringstream(line) >> lab >> std::ws >> stri >> std::ws >> strj >> std::ws >> strk;
            std::stringstream(stri.substr(0,stri.find('/'))) >> i;
            std::stringstream(strj.substr(0,strj.find('/'))) >> j;
            std::stringstream(strk.substr(0,strk.find('/'))) >> k;
            // The -1 operations are because C++ uses zero-based incides while wavefront uses one-based indexing
            faces_vec.push_back({vertices.at(i-1),vertices.at(j-1),vertices.at(k-1)});
          }
        }
      }
      // Store the domain
      domain_lo = {xl,yl,zl};
      domain_hi = {xh,yh,zh};
      // Write the faces vector to a YAKL array, move to device, and store in struct
      floatHost3d mesh_faces_host("faces",faces_vec.size(),3,3);
      for (int i=0; i < faces_vec.size(); i++) {
        mesh_faces_host(i,0,0) = faces_vec.at(i).v1.x;
        mesh_faces_host(i,0,1) = faces_vec.at(i).v1.y;
        mesh_faces_host(i,0,2) = faces_vec.at(i).v1.z;
        mesh_faces_host(i,1,0) = faces_vec.at(i).v2.x;
        mesh_faces_host(i,1,1) = faces_vec.at(i).v2.y;
        mesh_faces_host(i,1,2) = faces_vec.at(i).v2.z;
        mesh_faces_host(i,2,0) = faces_vec.at(i).v3.x;
        mesh_faces_host(i,2,1) = faces_vec.at(i).v3.y;
        mesh_faces_host(i,2,2) = faces_vec.at(i).v3.z;
      }
      this->faces = mesh_faces_host.createDeviceCopy();
      file.close();
    }

void TriMesh::add_offset(float x , float y , float z ) {
      YAKL_SCOPE( faces , this->faces );
      yakl::parallel_for( YAKL_AUTO_LABEL() , faces.extent(0) , KOKKOS_LAMBDA (int i) {
        faces(i,0,0) += x;    faces(i,0,1) += y;    faces(i,0,2) += z;
        faces(i,1,0) += x;    faces(i,1,1) += y;    faces(i,1,2) += z;
        faces(i,2,0) += x;    faces(i,2,1) += y;    faces(i,2,2) += z;
      });
      domain_lo.x += x;    domain_lo.y += y;    domain_lo.z += z;
      domain_hi.x += x;    domain_hi.y += y;    domain_hi.z += z;
    }

void TriMesh::apply_scaling(float sx, float sy, float sz) {
      YAKL_SCOPE( faces , this->faces );
      yakl::parallel_for( YAKL_AUTO_LABEL() , faces.extent(0) , KOKKOS_LAMBDA (int i) {
        faces(i,0,0) *= sx;    faces(i,0,1) *= sy;    faces(i,0,2) *= sz;
        faces(i,1,0) *= sx;    faces(i,1,1) *= sy;    faces(i,1,2) *= sz;
        faces(i,2,0) *= sx;    faces(i,2,1) *= sy;    faces(i,2,2) *= sz;
      });
      domain_lo.x *= sx;    domain_lo.y *= sy;    domain_lo.z *= sz;
      domain_hi.x *= sx;    domain_hi.y *= sy;    domain_hi.z *= sz;
    }

void TriMesh::zero_domain_lo() { add_offset( -domain_lo.x , -domain_lo.y , -domain_lo.z ); }

std::ostream &operator<<(std::ostream& os, TriMesh const &m ) {
      std::cout << "Bounding Box:    " << m.domain_lo << " x " << m.domain_hi << "\n";
      std::cout << "Number of faces: " << m.faces.extent(0) << std::endl;
      return os;
    }

} // namespace modules
