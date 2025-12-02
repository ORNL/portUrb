
#pragma once

#include "main_header.h"

namespace core {

  // This is a template class to hold multiple fields of the same type T
  //  up to a maximum of MAX_FIELDS fields.
  // This is useful for holding multiple tracer fields or other similar variables
  // The class provides methods to add fields, get fields, and access field data
  //  using operator() overloading for easy indexing
  // The fields are stored in a yakl::SArray for efficient access on device or host
  // The class supports copy and move semantics for easy passing and returning
  // The overloaded operator() allows accessing elements of each field with the appropriate number of indices
  //  as if accessing a yakl::Array directly with an additional leading index for the field index as if this
  //  were a YAKL array of one dimension higher.
  template <int MAX_FIELDS, class T>
  class MultipleFields {
  public:
    yakl::SArray<T,1,MAX_FIELDS> fields;  // SArray to hold up to MAX_FIELDS fields of type T
    int num_fields;                       // Number of fields currently stored

    // Default constructor to initialize num_fields to zero
    KOKKOS_INLINE_FUNCTION MultipleFields() { num_fields = 0; }

    // Copy constructor
    KOKKOS_INLINE_FUNCTION MultipleFields(MultipleFields const &rhs) {
      this->num_fields = rhs.num_fields;
      for (int i=0; i < num_fields; i++) {
        this->fields(i) = rhs.fields(i);
      }
    }

    // Copy assignment operator
    KOKKOS_INLINE_FUNCTION MultipleFields & operator=(MultipleFields const &rhs) {
      this->num_fields = rhs.num_fields;
      for (int i=0; i < num_fields; i++) {
        this->fields(i) = rhs.fields(i);
      }
      return *this;
    }

    // Move constructor
    KOKKOS_INLINE_FUNCTION MultipleFields(MultipleFields &&rhs) {
      this->num_fields = rhs.num_fields;
      for (int i=0; i < num_fields; i++) {
        this->fields(i) = rhs.fields(i);
      }
    }

    // Move assignment operator
    KOKKOS_INLINE_FUNCTION MultipleFields& operator=(MultipleFields &&rhs) {
      this->num_fields = rhs.num_fields;
      for (int i=0; i < num_fields; i++) {
        this->fields(i) = rhs.fields(i);
      }
      return *this;
    }

    // Add a new field to the MultipleFields object. This must match the type T.
    // field : The field to add
    // Increments num_fields by one
    KOKKOS_INLINE_FUNCTION void add_field( T field ) {
      this->fields(num_fields) = field;
      num_fields++;
    }

    // Get a field by its index
    // tr : Index of the field to get (0-based)
    // Returns the field at the specified index
    KOKKOS_INLINE_FUNCTION T &get_field( int tr ) const {
      return this->fields(tr);
    }

    // Get the number of fields currently stored
    KOKKOS_INLINE_FUNCTION int get_num_fields() const { return num_fields; }
    // Get the size (number of fields). Same as get_num_fields()
    KOKKOS_INLINE_FUNCTION int size          () const { return num_fields; }

    // Overloaded operator() to access elements of each field
    // tr : Index of the field to access (0-based)
    // i1,i2,... : Indices to access within the field
    // Returns a reference to the element at the specified indices within the specified field
    // The number of indices must match the rank of the field T, which is assumed to be a yakl::Array
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1) const ->
                                decltype(fields(tr)(i1)) {
      return this->fields(tr)(i1);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2) const ->
                                decltype(fields(tr)(i1,i2)) {
      return this->fields(tr)(i1,i2);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3) const ->
                                decltype(fields(tr)(i1,i2,i3)) {
      return this->fields(tr)(i1,i2,i3);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4) const ->
                                decltype(fields(tr)(i1,i2,i3,i4)) {
      return this->fields(tr)(i1,i2,i3,i4);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4, int i5) const ->
                                decltype(fields(tr)(i1,i2,i3,i4,i5)) {
      return this->fields(tr)(i1,i2,i3,i4,i5);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4, int i5, int i6) const ->
                                decltype(fields(tr)(i1,i2,i3,i4,i5,i6)) {
      return this->fields(tr)(i1,i2,i3,i4,i5,i6);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4, int i5, int i6, int i7) const ->
                                decltype(fields(tr)(i1,i2,i3,i4,i5,i6,i7)) {
      return this->fields(tr)(i1,i2,i3,i4,i5,i6,i7);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) const ->
                                decltype(fields(tr)(i1,i2,i3,i4,i5,i6,i7,i8)) {
      return this->fields(tr)(i1,i2,i3,i4,i5,i6,i7,i8);
    }
  };


  // Alias for MultipleFields with maximum number of fields set to max_fields from main_header.h
  // T is the underlying type of each yakl::Array field
  // N is the rank of each yakl::Array field
  // This is in device memory space with C-style indexing
  template <class T, int N>
  using MultiField = MultipleFields< max_fields , Array<T,N,memDevice,styleC> >;
}


