
#pragma once

#include "main_header.h"
#include <typeinfo>
#include <variant>
#include <string>
#include <vector>
#include <unordered_map>


namespace core {

  using yakl::Array;


  template <class memSpace = yakl::DeviceSpace>
  class DataManager {
  public:

    // This struct holds information about a single data entry
    template <class T> struct Entry {
      T *               ptr;        // Pointer to the allocated data
      size_t            bytes;      // Size of the allocated data in bytes
      std::vector<int>  dims;       // Dimensions of the data entry
      bool              positive;   // Whether the data is constrained to be positive
      bool              dirty;      // Whether the data has been modified since last reset of dirty flag
    };

    using EntryValue = std::variant< Entry<short int             > ,
                                     Entry<int                   > ,
                                     Entry<long int              > ,
                                     Entry<long long int         > ,
                                     Entry<unsigned short int    > ,
                                     Entry<unsigned int          > ,
                                     Entry<unsigned long int     > ,
                                     Entry<unsigned long long int> ,
                                     Entry<float                 > ,
                                     Entry<double                > ,
                                     Entry<long double           > ,
                                     Entry<bool                  > ,
                                     Entry<char                  > ,
                                     Entry<unsigned char         > >;

    bool static constexpr on_device = std::is_same_v<memSpace,yakl::DeviceSpace>;

    std::unordered_map<std::string,EntryValue> entries;

    DataManager()                                   = default;
    DataManager( DataManager &&rhs)                 = default;
    DataManager &operator=( DataManager &&rhs)      = default;
    DataManager( DataManager const &dm )            = delete;
    DataManager &operator=( DataManager const &dm ) = delete;
    ~DataManager() { finalize(); }


    
    void clone_into( DataManager<memSpace> & dm ) const {
      Kokkos::fence();
      for (auto & [name,entry] : entries) {
        std::visit( [&]<typename T>(Entry<T> const & entry) {
          dm.register_and_allocate<T>( name , entry.dims , entry.positive );
          this->get_collapsed<T const>(name).deep_copy_to(dm.get_collapsed<T>(name));
          dm.clean_entry(name);
        } , entry );
      }
      Kokkos::fence();
    }



    template <class T>
    void register_and_allocate( std::string      name             ,
                                std::vector<int> dims             ,
                                bool             positive = false ) {
      if (name == "") throw std::runtime_error("ERROR: You cannot register_and_allocate with an empty name");
      auto it = entries.find(name);
      if (it != entries.end()) {
        std::visit( [&](auto & e) {
          if (dims != e.dims)         throw std::runtime_error(std::string("ERROR: Trying to re-register name [")+
                                                               name+std::string("] with different dimensions"));
          if (positive != e.positive) throw std::runtime_error(std::string("ERROR: Trying to re-register name [")+
                                                               name+std::string("] with different positivity attribute"));
        } , it->second );
        return;
      }
      Entry<T> loc;
      if constexpr (on_device) { loc.ptr = (T *) yakl::alloc_device( get_data_size(dims)*sizeof(T) , name.c_str() ); }
      else                     { loc.ptr = (T *) ::malloc( get_data_size(dims)*sizeof(T) ); }
      loc.bytes     = get_data_size(dims)*sizeof(T);
      loc.dims      = dims;
      loc.positive  = positive;
      loc.dirty     = false;
      entries[std::move(name)] = loc;
    }


    void unregister_and_deallocate( std::string name ) {
      auto it = entries.find(name);
      if (it == entries.end()) return;
      std::visit( [&](auto & e) {
        if constexpr (on_device) { yakl::free_device(e.ptr, name.c_str()); }
        else                     { ::free(e.ptr);                          }
      } , it->second );
      entries.erase( name );
    }


    void clean_all_entries() {
      for (auto & [name,entry] : entries) {
        std::visit( [](auto & e) { e.dirty = false; } , entry );
      }
    }


    void clean_entry( std::string name ) {
      auto it = entries.find(name);
      if (it == entries.end()) return;
      std::visit( [](auto & e) { e.dirty = false; } , it->second );
    }


    bool entry_is_dirty(std::string name) const {
      auto it = entries.find(name);
      if (it == entries.end()) return false;
      return std::visit( [](auto & e) { return e.dirty; } , it->second );
    }


    std::vector<std::string> get_dirty_entries( ) const {
      std::vector<std::string> dirty_entries;
      for (auto & [name,entry] : entries) {
        std::visit( [&](auto & e) { if (e.dirty) dirty_entries.push_back(name); } , entry );
      }
      return dirty_entries;
    }


    bool entry_exists( std::string name ) const { return entries.find(name) != entries.end(); }


    template <class T, int N> requires std::is_const_v<T>
    Array<typename yakl::ViewType<T,N>::type,memSpace> get( std::string name ) const {
      using TNC = std::remove_const_t<T>;
      auto it = entries.find(name);
      if (it == entries.end()) throw std::runtime_error(std::string("Entry not found: ")+name);
      auto & entry = std::get<Entry<TNC>>(it->second);
      if (entry.dims.size() != N) throw std::runtime_error(std::string("ERROR: Calling get() with name [")+
                                                           name+std::string("] with the wrong number of dimensions"));
      // Return an unmanaged yakl::Array that wraps the entry's data pointer and dimensions
      return [&] <std::size_t... Is> (std::index_sequence<Is...>) {
        return Array<typename yakl::ViewType<T,N>::type,memSpace>( static_cast<T *>(entry.ptr) , entry.dims[Is]... );
      } (std::make_index_sequence<N>{});
    }


    template <class T, int N>  requires (!std::is_const_v<T>)
    Array<typename yakl::ViewType<T,N>::type,memSpace> get( std::string name ) {
      auto it = entries.find(name);
      if (it == entries.end()) throw std::runtime_error(std::string("Entry not found: ")+name);
      auto & entry = std::get<Entry<T>>(it->second);
      if (entry.dims.size() != N) throw std::runtime_error(std::string("ERROR: Calling get() with name [")+
                                                           name+std::string("] with the wrong number of dimensions"));
      entry.dirty = true;
      // Return an unmanaged yakl::Array that wraps the entry's data pointer and dimensions
      return [&] <std::size_t... Is> (std::index_sequence<Is...>) {
        return Array<typename yakl::ViewType<T,N>::type,memSpace>( static_cast<T *>(entry.ptr) , entry.dims[Is]... );
      } (std::make_index_sequence<N>{});
    }


    template <class T> requires std::is_const_v<T>
    Array<T **,memSpace> get_lev_col( std::string name ) const {
      using TNC = std::remove_const_t<T>;
      auto it = entries.find(name);
      if (it == entries.end()) throw std::runtime_error(std::string("Entry not found: ")+name);
      auto & entry = std::get<Entry<TNC>>(it->second);
      if (entry.dims.size() < 2) throw std::runtime_error(std::string("ERROR: Calling get_lev_col() with name [")+
                                                          name+std::string("] with the wrong number of dimensions"));
      int nlev = entry.dims.at(0); // First dimension is assumed to be vertical levels
      int ncol = 1;                // All other dimensions are collapsed into a single horizontal dimension
      for (int i=1; i < entry.dims.size(); i++) { ncol *= entry.dims.at(i); }
      return Array<T **,memSpace>( static_cast<T *>(entry.ptr) , nlev , ncol );
    }


    template <class T> requires (!std::is_const_v<T>)
    Array<T **,memSpace> get_lev_col( std::string name ) {
      auto it = entries.find(name);
      if (it == entries.end()) throw std::runtime_error(std::string("Entry not found: ")+name);
      auto & entry = std::get<Entry<T>>(it->second);
      if (entry.dims.size() < 2) throw std::runtime_error(std::string("ERROR: Calling get_lev_col() with name [")+
                                                          name+std::string("] with the wrong number of dimensions"));
      entry.dirty = true;
      int nlev = entry.dims.at(0); // First dimension is assumed to be vertical levels
      int ncol = 1;                         // All other dimensions are collapsed into a single horizontal dimension
      for (int i=1; i < entry.dims.size(); i++) { ncol *= entry.dims.at(i); }
      return Array<T **,memSpace>( static_cast<T *>(entry.ptr) , nlev , ncol );
    }


    template <class T> requires std::is_const_v<T>
    Array<T *,memSpace> get_collapsed( std::string name ) const {
      using TNC = std::remove_const_t<T>;
      auto it = entries.find(name);
      if (it == entries.end()) throw std::runtime_error(std::string("Entry not found: ")+name);
      auto & entry = std::get<Entry<TNC>>(it->second);
      int ncells = entry.dims.at(0); // number of elements is the product of all dimensions
      for (int i=1; i < entry.dims.size(); i++) { ncells *= entry.dims.at(i); }
      return Array<T *,memSpace>( static_cast<T *>(entry.ptr) , ncells );
    }


    template <class T> requires (!std::is_const_v<T>)
    Array<T *,memSpace> get_collapsed( std::string name ) {
      auto it = entries.find(name);
      if (it == entries.end()) throw std::runtime_error(std::string("Entry not found: ")+name);
      auto & entry = std::get<Entry<T>>(it->second);
      entry.dirty = true;
      int ncells = entry.dims.at(0); // number of elements is the product of all dimensions
      for (int i=1; i < entry.dims.size(); i++) { ncells *= entry.dims.at(i); }
      return Array<T *,memSpace>( static_cast<T *>(entry.ptr) , ncells );
    }


    int get_data_size( std::vector<int> const & dims ) const {
      int size = 1;
      for (int i=0; i < dims.size(); i++) { size *= dims.at(i); }
      return size;
    }


    void validate_all( bool die = false , std::string file = "" , int line = 0 , int rank = 0 ) const {
      for (auto const & [name,entry] : entries) { validate(name,die,file,line,rank); }
    }


    void validate( std::string name , bool die = false , std::string file = "" , int line = 0 , int rank = 0 ) const {
      auto it = entries.find(name);
      if (it == entries.end()) return;
      std::visit( [&]<typename T>(Entry<T> const & entry) {
        auto v = get_collapsed<T const>(name);
        bool nan_found = false;
        bool inf_found = false;
        bool neg_found = false;
        if constexpr (std::is_floating_point_v<T>) {
          nan_found = yakl::intrinsics::any( yakl::componentwise::isnan(v) );
          inf_found = yakl::intrinsics::any( yakl::componentwise::isinf(v) );
        }
        if constexpr (!std::is_same_v<T,bool> && !std::is_unsigned_v<T>) {
          if (entry.positive) {
            using yakl::componentwise::operator<;
            neg_found = yakl::intrinsics::any( v < 0 );
          }
        }
        if (nan_found) std::cerr << "WARNING:" << file << ":" << line << ":" << rank << ": NaN in entry " << name << std::endl;
        if (inf_found) std::cerr << "WARNING:" << file << ":" << line << ":" << rank << ": inf in entry " << name << std::endl;
        if (neg_found) std::cerr << "WARNING:" << file << ":" << line << ":" << rank << ": neg in entry " << name << std::endl;
        if (die && (nan_found || inf_found || neg_found)) throw std::runtime_error("");
      } , it->second );
    }


    void finalize() {
      Kokkos::fence();
      for (auto & [name,entry] : entries) {
        std::visit( [&] (auto & e) {
          if constexpr (on_device) { yakl::free_device(e.ptr, name.c_str()); }
          else                     { ::free(e.ptr);                          }
        } , entry );
      }
      entries.clear();
    }

  };


  // Host specialization of the DataManager to manage data in host memory
  typedef DataManager<Kokkos::HostSpace> DataManagerHost;

}


