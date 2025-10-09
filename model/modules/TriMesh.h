
#pragma once

#include "coupler.h"

namespace modules {

  // Holds a set of triangular faces defined in 3-D. Reads wavefront .obj files defined with faces that are
  // purely triangular. Provides a convenience function to generate a heightmap of the highest point over
  // a grid among all of the stored faces.
  struct TriMesh {

    struct Vertex {
      float x, y, z;
      inline friend std::ostream &operator<<(std::ostream& os, Vertex const &v ) {
        os << "[" << v.x << " , " << v.y << " , " << v.z << "]";
        return os;
      }
    };

    struct Face {
      Vertex v1, v2, v3;
    };


    float3d faces;
    Vertex  domain_lo;
    Vertex  domain_hi;


    void load_file(std::string fname) {
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


    // Add an offset to all face vertices and the domain extents. This is typically used to set lower bounds
    // to zero.
    void add_offset(float x = 0, float y = 0, float z = 0) {
      YAKL_SCOPE( faces , this->faces );
      yakl::c::parallel_for( YAKL_AUTO_LABEL() , faces.extent(0) , KOKKOS_LAMBDA (int i) {
        faces(i,0,0) += x;    faces(i,0,1) += y;    faces(i,0,2) += z;
        faces(i,1,0) += x;    faces(i,1,1) += y;    faces(i,1,2) += z;
        faces(i,2,0) += x;    faces(i,2,1) += y;    faces(i,2,2) += z;
      });
      domain_lo.x += x;    domain_lo.y += y;    domain_lo.z += z;
      domain_hi.x += x;    domain_hi.y += y;    domain_hi.z += z;
    }


    // Set domain_lo to zero
    void zero_domain_lo() { add_offset( -domain_lo.x , -domain_lo.y , -domain_lo.z ); }


    // Create a heightmap of the covered domain extent using the defined grid spacing. This is defined as
    // the maximum height over all faces for each point in a grid.
    KOKKOS_INLINE_FUNCTION static float max_height(float x, float y, float3d const &faces_in, float domain_lo_z) {
      float ret = domain_lo_z;
      for (int k=0; k < faces_in.extent(0); k++) { ret = std::max( ret , interp(faces_in,k,x,y,domain_lo_z) ); }
      return ret;
    }


    // Interpolate the height of the given horizontal point location using surrounding face data
    KOKKOS_INLINE_FUNCTION static float interp( float3d const &faces_in, int k, float x, float y, float domain_lo_z ) {
      double constexpr eps = 1.e-8;
      double v1_x = faces_in(k,0,0);    double v1_y = faces_in(k,0,1);    double v1_z = faces_in(k,0,2);
      double v2_x = faces_in(k,1,0);    double v2_y = faces_in(k,1,1);    double v2_z = faces_in(k,1,2);
      double v3_x = faces_in(k,2,0);    double v3_y = faces_in(k,2,1);    double v3_z = faces_in(k,2,2);
      // Area of the triangle
      double area = 0.5 * (v1_x*(v2_y - v3_y) + v2_x*(v3_y - v1_y) + v3_x*(v1_y - v2_y));
      if (std::abs(area) < eps) return domain_lo_z;
      // Interpolation weights
      double w1 = (v2_x*v3_y - v3_x*v2_y + (v2_y - v3_y)*x + (v3_x - v2_x)*y) / (2*area);
      double w2 = (v3_x*v1_y - v1_x*v3_y + (v3_y - v1_y)*x + (v1_x - v3_x)*y) / (2*area);
      double w3 = 1 - w1 - w2;
      // Interpolate z value if weights in [0,1] (i.e., the point's within this triangle's horizontal area)
      if (w1>=-eps && w2>=-eps && w3>=-eps && w1<=1+eps && w2<=1+eps && w3<=1+eps) { return w1*v1_z + w2*v2_z + w3*v3_z; }
      else                                                                         { return domain_lo_z;                 }
    }


    // Tell the user a bit about this set of faces
    inline friend std::ostream &operator<<(std::ostream& os, TriMesh const &m ) {
      std::cout << "Bounding Box:    " << m.domain_lo << " x " << m.domain_hi << "\n";
      std::cout << "Number of faces: " << m.faces.extent(0) << std::endl;
      return os;
    }

  };

}

