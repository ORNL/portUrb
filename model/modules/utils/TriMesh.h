
#pragma once

#include "coupler.h"

namespace modules {

  // Holds a set of triangular faces defined in 3-D. Reads wavefront .obj files defined with faces that are
  // purely triangular. Provides a convenience function to generate a heightmap of the highest point over
  // a grid among all of the stored faces.
  struct TriMesh {

    // This class holds data to define a vertex
    struct Vertex {
      float x, y, z;
      friend std::ostream &operator<<(std::ostream& os, Vertex const &v );
    };

    // This class holds data to define a triangular face using three vertices
    struct Face {
      Vertex v1, v2, v3;
    };


    float3d faces;      // YAKL array holding the triangular faces. Dimensions are (num_faces,3,3)
    Vertex  domain_lo;  // Lower extent of the domain containing all faces
    Vertex  domain_hi;  // Upper extent of the domain containing all faces


    // Load a wavefront .obj file containing triangular vertices and faces. Iterate through the file line by line,
    //   storing vertices and faces as they are encountered, keeping track of the domain extents as vertices are read.
    void load_file(std::string fname);


    // Add an offset to all face vertices and the domain extents. This is typically used to set lower bounds
    // to zero.
    void add_offset(float x = 0, float y = 0, float z = 0);


    // Add an offset to all face vertices and the domain extents. This is typically used to set lower bounds
    // to zero.
    void apply_scaling(float sx, float sy, float sz);


    // Set domain_lo to zero
    void zero_domain_lo();


    // Create a heightmap of the covered domain extent using the defined grid spacing. This is defined as
    // the maximum height over all faces for each point in a grid.
    KOKKOS_INLINE_FUNCTION static float max_height(float x, float y, float3d const &faces_in, float domain_lo_z) {
      float ret = domain_lo_z;
      for (int k=0; k < faces_in.extent(0); k++) { ret = std::max( ret , interp(faces_in,k,x,y,domain_lo_z) ); }
      return ret;
    }


    // Interpolate the height of the given horizontal point location using surrounding face data
    KOKKOS_INLINE_FUNCTION static float interp( float3d const &faces_in, int k, float x, float y, float domain_lo_z ) {
      double constexpr eps = 1.e-10;
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
    friend std::ostream &operator<<(std::ostream& os, TriMesh const &m );

  };

}

