
#pragma once

#include "main_header.h"
#include <typeinfo>


namespace core {

  using yakl::Array;

  // The mutex is used to protect DataManager operations that modify shared data structures
  //  such as registering and allocating entries from concurrent access in multi-threaded contexts
  extern std::mutex data_manager_mutex;


  // The DataManagerTemplate class manages named data entries and dimensions in a specified memory space
  // It allows registering, allocating, deallocating, and accessing data entries with associated metadata
  // The class supports both device and host memory spaces based on the memSpace template parameter.
  // The class provides methods to add dimensions, register and allocate entries, find entries and dimensions,
  //  finalize the data manager by deallocating all entries, and clone the data manager into another instance.
  // The class uses function pointers for allocation and deallocation to support different memory spaces.
  // The class is designed to be thread-safe for concurrent access in multi-threaded environments.
  // The class does not support copy semantics to avoid accidental copying of large data structures.
  // The class supports move semantics for efficient transfer of ownership of data structures.
  // The class uses yakl::alloc_device and yakl::free_device for device memory management
  //  and standard malloc and free for host memory management.
  // The class uses yakl::memcpy_device_to_device_void and yakl::memcpy_host_to_host_void for data copying
  //  during cloning operations.
  // The class provides detailed error messages and terminates the program on errors.
  // Typically, all allocateed variables in a simulation will be managed by a DataManager object.
  // The DataManagerTemplate class is typically not used directly; instead, typedefs for
  //  DataManagerDevice and DataManagerHost are used for device and host memory spaces respectively
  //  (see the typedefs at the end of this file).
  template <int memSpace = yakl::memDevice>
  class DataManagerTemplate {
  public:

    // This struct holds information about a single data entry
    struct Entry {
      std::string              name;       // Unique name of the data entry
      std::string              desc;       // Description of the data entry
      size_t                   type_hash;  // Hash of the data type for type checking for get methods
      void *                   ptr;        // Pointer to the allocated data
      size_t                   bytes;      // Size of the allocated data in bytes
      std::vector<int>         dims;       // Dimensions of the data entry
      std::vector<std::string> dim_names;  // Names of each dimension (must match dimensions lengths in dims)
      bool                     positive;   // Whether the data is constrained to be positive
      bool                     dirty;      // Whether the data has been modified since last reset of dirty flag
    };

    // Holds information about a single dimension
    struct Dimension {
      std::string name; // Name of the dimension
      int         len;  // Length of the dimension
    };

    std::vector<Entry>     entries;     // Vector of all registered data entries
    std::vector<Dimension> dimensions;  // Vector of all registered dimensions

    std::function<void *( size_t , char const * )> allocate;    // Function to allocate memory in the specified memory space
    std::function<void  ( void * , char const * )> deallocate;  // Function to deallocate memory in the specified memory space

    int num_assigned_dims;              // Number of dimensions that have been assigned to entries


    // Constructor initializes the DataManagerTemplate object
    // memSpace : Memory space for the DataManager (memDevice or memHost)
    // Sets the allocate and deallocate function pointers based on the memSpace template parameter
    // Initializes entries and dimensions as empty vectors
    // Typically, a DataManagerTemplate object is created for either device or host memory space
    //  and used to manage all allocated variables in that memory space
    // The constructor does not allocate any memory; memory is allocated when entries are registered
    //  and allocated using the register_and_allocate method
    DataManagerTemplate() {
      entries    = std::vector<Entry>();
      dimensions = std::vector<Dimension>();
      num_assigned_dims = 0;
      if (memSpace == memDevice) {
        allocate   = [] (size_t bytes,char const *label) -> void * { return yakl::alloc_device(bytes,label); };
        deallocate = [] (void *ptr   ,char const *label)           {        yakl::free_device (ptr  ,label); };
      } else if (memSpace == memHost) {
        allocate   = [] (size_t bytes,char const *label) -> void * { return ::malloc(bytes); };
        deallocate = [] (void *ptr   ,char const *label)           {        ::free  (ptr); };
      } else {
        Kokkos::abort("ERROR: DataManagerTemplate created with invalid memSpace template parameter");
      }
    }

    // Move constructor
    DataManagerTemplate( DataManagerTemplate &&rhs) = default;

    // Move assignment operator
    DataManagerTemplate &operator=( DataManagerTemplate &&rhs) = default;

    // Delete copy constructor and copy assignment operator to avoid accidental copying of large data structures
    DataManagerTemplate( DataManagerTemplate const &dm ) = delete;

    // Delete copy assignment operator to avoid accidental copying of large data structures
    DataManagerTemplate &operator=( DataManagerTemplate const &dm ) = delete;

    // Destructor finalizes the DataManagerTemplate object by deallocating all entries
    ~DataManagerTemplate() {
      // finalize deallocates all entries and resets entries and dimensions to empty vectors
      finalize();
    }


    
    // Clone this DataManagerTemplate into another DataManagerTemplate instance
    // dm : DataManagerTemplate instance to clone into
    // The clone_into method allocates new memory for each entry in the target DataManagerTemplate
    //  and copies the data from this DataManagerTemplate into the target instance
    // The dimensions vector is also copied to the target instance
    // The allocate and deallocate function pointers of the target instance are not modified
    // This method is useful for creating a copy of the DataManagerTemplate in another memory space
    //  or for creating a separate instance with the same data
    // The method uses yakl::memcpy_device_to_device_void or yakl::memcpy_host_to_host_void
    //  for copying data based on the memSpace template parameter
    // The method assumes that the target DataManagerTemplate is empty before cloning into it
    void clone_into( DataManagerTemplate<memSpace> &dm ) const {
      // Copy the allocate, deallocate, dimensions, and num_assigned_dims members
      dm.allocate          = this->allocate;
      dm.deallocate        = this->deallocate;
      dm.dimensions        = this->dimensions;
      dm.num_assigned_dims = this->num_assigned_dims;

      // Clone each entry into the target DataManagerTemplate object
      for (auto &entry : this->entries) {
        Entry loc; // Local Entry to hold cloned data
        // Copy the name, desc, and type_hash
        loc.name      = entry.name;
        loc.desc      = entry.desc;
        loc.type_hash = entry.type_hash;
        // Allocate new memory for the entry in the target DataManagerTemplate
        loc.ptr       = allocate( entry.bytes , entry.name.c_str() );
        // Copy the data from this DataManagerTemplate into the target instance
        if (memSpace == yakl::memHost) {
          yakl::memcpy_host_to_host_void    ( loc.ptr , entry.ptr , entry.bytes );
        } else {
          yakl::memcpy_device_to_device_void( loc.ptr , entry.ptr , entry.bytes );
          Kokkos::fence();
        }
        // Copy bytes, dims, dim_names, positive, and dirty flag
        loc.bytes     = entry.bytes;
        loc.dims      = entry.dims;
        loc.dim_names = entry.dim_names;
        loc.positive  = entry.positive;
        loc.dirty     = entry.dirty;
        dm.entries.push_back(loc); // Add the cloned entry to the target DataManagerTemplate
      }
    }


    // Add a dimension with the given name and length
    // name : Name of the dimension to add
    // len  : Length of the dimension to add
    // If a dimension with the same name already exists, checks that the length matches
    //  and does not add a duplicate entry
    // If the length does not match, prints an error message and terminates the program
    // This method is useful for defining dimensions that can be referenced by entries
    //  when registering and allocating data entries with the register_and_allocate method
    void add_dimension( std::string name , int len ) {
      int dimid = find_dimension( name );
      if (dimid > 0) { // If the dimension already exists, check that the length matches
        if ( dimensions.at(dimid).len != len ) {
          std::cerr << "ERROR: Attempting to add a dimension of name [" << name
                    << "] with length [" << len 
                    << "]. However, it already exists with length [" << dimensions.at(dimid).len << "].";
          endrun();
        }
        return;  // Avoid adding a duplicate entry
      }
      dimensions.push_back( {name , len} ); // 
    }


    // Create an entry and allocate it. if dim_names is passed, then check dimension sizes for consistency
    // if positive == true, then positivity validation checks for positivity; otherwise, ignores it.
    // While zeroing the allocation upon creation might be nice, it's not efficient in all GPU contexts
    // because many separate kernels are more expensive than one big one when data sizes are small.
    // So it's up to the user to zero out the arrays they allocate to make valgrind happy and avoid unhappy
    // irreproducible bugs.

    // The register_and_allocate method registers a new data entry with the DataManagerTemplate
    //  and allocates memory for it based on the provided dimensions
    // T        : Template parameter for the data type of the entry to register and allocate
    // name      : Name of the entry to register and allocate
    // desc      : Description of the entry
    // dims      : Vector of dimension lengths for the entry
    // dim_names : (optional) Vector of dimension names for the entry
    // positive  : (optional) Whether the data is constrained to be positive (default: false)
    // If dim_names is provided, checks that the lengths match the provided dims
    // If a dimension name does not exist, it is added to the dimensions vector
    // If a dimension name exists, checks that the length matches the provided dims
    // If the lengths do not match, prints an error message and terminates the program
    // If an entry with the same name already exists, checks that the attributes match
    //  and does not add a duplicate entry
    // If the attributes do not match, prints an error message and terminates the program 
    // Allocates memory for the entry using the allocate function pointer
    template <class T>
    void register_and_allocate( std::string name ,
                                std::string desc ,
                                std::vector<int> dims ,
                                std::vector<std::string> dim_names = std::vector<std::string>() ,
                                bool positive = false ) {
      static std::mutex data_manager_mutex; // Mutex to protect concurrent access
      // If the name is empty, print an error and terminate
      if (name == "") {
        endrun("ERROR: You cannot register_and_allocate with an empty string");
      }
      // Make sure we don't have a duplicate entry
      int entry_ind = find_entry(name);
      if ( entry_ind != -1) { // If the entry already exists, check that the attributes match
        if (dims != entries.at(entry_ind).dims) { // If dims do not match, print error and terminate
          std::cerr << "ERROR: Trying to re-register name [" << name << "] with different dimensions";
          endrun();
        }
        if (dim_names != entries.at(entry_ind).dim_names) { // If dim_names do not match, print error and terminate
          std::cerr << "ERROR: Trying to re-register name [" << name << "] with different dimension names";
          endrun();
        }
        if (positive != entries.at(entry_ind).positive) { // if positive attribute does not match, print error and terminate
          std::cerr << "ERROR: Trying to re-register name [" << name << "] with different positivity attribute";
          endrun();
        }
        // Ignore the re-registration request if the attributes match
        return;
      }

      if (dim_names.size() > 0) { // If dim_names was passed, check that sizes match
        if (dims.size() != dim_names.size()) {
          std::cerr << "ERROR: Trying to register and allocate name [" << name << "]. ";
          endrun("Must have the same number of dims and dim_names");
        }
        // Make sure the dimensions are the same size as existing ones of the same name
        for (int i=0; i < dim_names.size(); i++) { // Loop through passed dimension names, and check sizes
          int dimid = find_dimension(dim_names.at(i));
          if (dimid == -1) {
            Dimension loc;
            loc.name = dim_names.at(i);
            loc.len  = dims     .at(i);
            dimensions.push_back(loc);
          } else {
            if (dimensions.at(dimid).len != dims.at(i)) {
              std::cerr << "ERROR: Trying to register and allocate name [" << name << "]. " <<
                           "Dimension of name [" << dim_names.at(i) << "] already exists with a different " <<
                           "length of [" << dimensions.at(dimid).len << "]. The length you provided for " << 
                           "that dimension name in this call is [" << dims.at(i) << "]. ";
              endrun("");
            }
          }
        }
      } else { // If dim_names was not passed, then create them with unique assigned names or use existing ones
        std::string loc_dim_name = "";
        for (int i=0; i < dims.size(); i++) { // i is local var dims index
          // Use existing dimension names if they match the length (first one found)
          // This is why x, y, and z dimensions are defined first in the coupler init() function
          for (int ii=0; ii < this->dimensions.size(); ii++) { // ii is data manager dimensions index
            if (dims.at(i) == this->dimensions.at(ii).len) loc_dim_name = this->dimensions.at(i).name;
            break;
          }
          // If no existing dimension name matches the length, create a new unique name
          if (loc_dim_name == "") {
            data_manager_mutex.lock();
            loc_dim_name = std::string("assigned_dim_") + std::to_string(this->num_assigned_dims);
            this->num_assigned_dims++;
            data_manager_mutex.unlock();
          }
          dim_names.push_back(loc_dim_name);
        }
      }

      // Create the local entry, and copy the parameters into it
      Entry loc;
      loc.name      = name;
      loc.desc      = desc;
      loc.type_hash = get_type_hash<T>();
      loc.ptr       = allocate( get_data_size(dims)*sizeof(T) , name.c_str() );
      loc.bytes     = get_data_size(dims)*sizeof(T);
      loc.dims      = dims;
      loc.dim_names = dim_names;
      loc.positive  = positive;
      loc.dirty     = false;

      entries.push_back( loc ); // Add the new entry to the entries vector
    }


    // Unregister and deallocate the entry with the given name
    // name : Name of the entry to unregister and deallocate
    // If the entry does not exist, prints an error message and terminates the program
    // This method deallocates the memory for the entry using the deallocate function pointer
    // This method is useful for cleaning up memory for entries that are no longer needed
    //  and for managing memory usage in the DataManagerTemplate
    // After deallocation, removes the entry from the entries vector
    void unregister_and_deallocate( std::string name ) {
      int id = find_entry_or_error( name );
      deallocate( entries.at(id).ptr , entries.at(id).name.c_str() );
      entries.erase( entries.begin() + id );
    }


    // Clean the dirty flag to false for all entries
    // when the dirty flag is true, then the entry has been potentially written to since its creation or previous cleaning
    void clean_all_entries() {
      for (int i=0; i < entries.size(); i++) { entries.at(i).dirty = false; }
    }


    // Clean the dirty flag to false for a single entry
    // when the dirty flag is true, then the entry has been potentially written to since its creation or previous cleaning
    void clean_entry( std::string name ) {
      int id = find_entry_or_error( name );
      entries.at(id).dirty = false;
    }


    // Determine if the entry with the given name is dirty
    // when the dirty flag is true, then the entry has been potentially written to since its creation or previous cleaning
    bool entry_is_dirty( std::string name ) const {
      int id = find_entry_or_error( name );
      return entries.at(id).dirty;
    }


    // Get the names of all dirty entries as a vector of strings
    // when the dirty flag is true, then the entry has been potentially written to since its creation or previous cleaning
    std::vector<std::string> get_dirty_entries( ) const {
      std::vector<std::string> dirty_entries;
      for (int i=0; i < entries.size(); i++) {
        if (entries.at(i).dirty) dirty_entries.push_back( entries.at(i).name );
      }
      return dirty_entries;
    }


    // Determine if an entry with the given name exists
    // name : Name of the entry to check
    // Returns true if the entry exists, false otherwise 
    // This method is useful for checking if an entry has been registered before attempting to access it
    bool entry_exists( std::string name ) const {
      int id = find_entry(name);
      if (id >= 0) return true;
      return false;
    }


    // Get a READ ONLY YAKL array (styleC) for the entry of this name
    // T must match the registered type (const and volatile are ignored in this comparison)
    // N must match the registered number of dimensions
    // This template specialization is for const T to enforce read-only access
    // If T is const, the dirty flag is not modified
    // name : Name of the entry to get
    // Returns a yakl::Array of type T and rank N for the entry with the given name
    // If the entry does not exist, prints an error message and terminates the program
    // If the type or number of dimensions do not match, prints an error message and terminates the program
    template <class T, int N , typename std::enable_if< std::is_const<T>::value , int >::type = 0 >
    Array<T,N,memSpace,styleC> get( std::string name ) const {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      // Make sure it's the right type and dimensionality
      if (!validate_type<T>(id)) {
        std::cerr << "ERROR: Calling get() with name [" << name << "] with the wrong type"; endrun("");
      }
      if (!validate_dims<N>(id)) {
        std::cerr << "ERROR: Calling get() with name [" << name << "] with the wrong number of dimensions"; endrun("");
      }
      // Create an unmanaged yakl::Array that wraps the entry's data pointer and dimensions
      Array<T,N,memSpace,styleC> ret( name.c_str() , (T *) entries.at(id).ptr , entries.at(id).dims );
      // Return the yakl::Array
      return ret;
    }


    // Get a READ/WRITE YAKL array (styleC) for the entry of this name
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // N must match the registered number of dimensions
    // This template specialization is for non-const T to allow read-write access
    // name : Name of the entry to get
    // Returns a yakl::Array of type T and rank N for the entry with the given name
    // If the entry does not exist, prints an error message and terminates the program
    // If the type or number of dimensions do not match, prints an error message and terminates the program
    // Additionally, sets the dirty flag to true for the entry because it isn't const
    // Use of this routine rather than the const-type routine indicates that the data may be modified
    template <class T, int N , typename std::enable_if< ! std::is_const<T>::value , int >::type = 0 >
    Array<T,N,memSpace,styleC> get( std::string name ) {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      entries.at(id).dirty = true;
      // Make sure it's the right type and dimensionality
      if (!validate_type<T>(id)) {
        std::cerr << "ERROR: Calling get() with name [" << name << "] with the wrong type"; endrun("");
      }
      if (!validate_dims<N>(id)) {
        std::cerr << "ERROR: Calling get() with name [" << name << "] with the wrong number of dimensions"; endrun("");
      }
      // Create an unmanaged yakl::Array that wraps the entry's data pointer and dimensions
      Array<T,N,memSpace,styleC> ret( name.c_str() , (T *) entries.at(id).ptr , entries.at(id).dims );
      // Return the yakl::Array
      return ret;
    }


    // This is the same as the other const get() method except it assumes the first dimension is vertical
    //  and that all other dimensions are horizontal and can be collapsed into a single horizontal dimension
    // T must match the registered type (const and volatile are ignored in this comparison)
    // This template specialization is for const T to enforce read-only access
    // If T is const, the dirty flag is not modified
    // name : Name of the entry to get
    // Returns a yakl::Array of type T and rank 2 for the entry with the given name
    // If the entry does not exist, prints an error message and terminates the program
    // If the number of dimensions is less than 2, prints an error message and terminates the program
    // The first dimension is treated as vertical levels, and all other dimensions are collapsed into a single horizontal dimension
    // Fastest varying dimensions in the aggregated horizontal dimensions are maintained
    // This method is useful for accessing data that is structured with vertical levels and horizontal columns
    template <class T, typename std::enable_if< std::is_const<T>::value , int>::type = 0 >
    Array<T,2,memSpace,styleC> get_lev_col( std::string name ) const {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      // Make sure it's the right type
      if (!validate_type<T>(id)) {
        std::cerr << "ERROR: Calling get_lev_col() with name [" << name << "] with the wrong type"; endrun("");
      }
      if (!validate_dims_lev_col(id)) {
        std::cerr << "ERROR: Calling get_lev_col() with name [" << name << "], but the variable's number of " <<
                     "dimensions is not compatible. You need two or more dimensions in the variable to call this.";
        endrun("");
      }
      int nlev = entries.at(id).dims.at(0); // First dimension is assumed to be vertical levels
      int ncol = 1;                         // All other dimensions are collapsed into a single horizontal dimension
      for (int i=1; i < entries.at(id).dims.size(); i++) {
        ncol *= entries.at(id).dims.at(i);
      }
      // Create an unmanaged yakl::Array that wraps the entry's data pointer and dimensions
      Array<T,2,memSpace,styleC> ret( name.c_str() , (T *) entries.at(id).ptr , nlev , ncol );
      // Return the yakl::Array
      return ret;
    }


    // Same as the previous get_lev_col method, but for non-const T to allow read-write access
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // name : Name of the entry to get
    // Returns a yakl::Array of type T and rank 2 for the entry with the given name
    // If the entry does not exist, prints an error message and terminates the program
    // If the number of dimensions is less than 2, prints an error message and terminates the program
    // The first dimension is treated as vertical levels, and all other dimensions are collapsed into a single horizontal dimension
    // Fastest varying dimensions in the aggregated horizontal dimensions are maintained
    // Additionally, sets the dirty flag to true for the entry because it isn't const
    // Use of this routine rather than the const-type routine indicates that the data may be modified
    template <class T, typename std::enable_if< ! std::is_const<T>::value , int>::type = 0 >
    Array<T,2,memSpace,styleC> get_lev_col( std::string name ) {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      entries.at(id).dirty = true;
      // Make sure it's the right type
      validate_type<T>(id);
      validate_dims_lev_col(id);
      int nlev = entries.at(id).dims.at(0); // First dimension is assumed to be vertical levels
      int ncol = 1;                         // All other dimensions are collapsed into a single horizontal dimension
      for (int i=1; i < entries.at(id).dims.size(); i++) {
        ncol *= entries.at(id).dims.at(i);
      }
      // Create an unmanaged yakl::Array that wraps the entry's data pointer and dimensions
      Array<T,2,memSpace,styleC> ret( name.c_str() , (T *) entries.at(id).ptr , nlev , ncol );
      // Return the yakl::Array
      return ret;
    }


    // Same as the previous get methods except assumes all dimensions are to be collapsed into a single dimension
    // T must match the registered type (const and volatile are ignored in this comparison)
    // This template specialization is for const T to enforce read-only access
    // If T is const, the dirty flag is not modified
    // name : Name of the entry to get
    // Returns a yakl::Array of type T and rank 1 for the entry with the given name
    // If the entry does not exist, prints an error message and terminates the program
    // All dimensions are collapsed to a single dimension.
    // Fastest varying dimensions in the aggregated dimensions are maintained.
    template <class T, typename std::enable_if< std::is_const<T>::value , int>::type = 0 >
    Array<T,1,memSpace,styleC> get_collapsed( std::string name ) const {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      // Make sure it's the right type
      validate_type<T>(id);
      int ncells = entries.at(id).dims.at(0); // number of elements is the product of all dimensions
      for (int i=1; i < entries.at(id).dims.size(); i++) {
        ncells *= entries.at(id).dims.at(i);
      }
      // Create an unmanaged yakl::Array that wraps the entry's data pointer and dimensions
      Array<T,1,memSpace,styleC> ret( name.c_str() , (T *) entries.at(id).ptr , ncells );
      // Return the yakl::Array
      return ret;
    }


    // Same as the previous get_collapsed method, but for non-const T to allow read-write access
    // If T is not const, then the dirty flag is set to true because it can be potentially written to
    // T must match the registered type (const and volatile are ignored in this comparison)
    // name : Name of the entry to get
    // Returns a yakl::Array of type T and rank 1 for the entry with the given name
    // If the entry does not exist, prints an error message and terminates the program
    // All dimensions are collapsed to a single dimension.
    // Fastest varying dimensions in the aggregated dimensions are maintained.
    template <class T, typename std::enable_if< ! std::is_const<T>::value , int>::type = 0 >
    Array<T,1,memSpace,styleC> get_collapsed( std::string name ) {
      // Make sure we have this name as an entry
      int id = find_entry_or_error( name );
      entries.at(id).dirty = true;
      // Make sure it's the right type
      validate_type<T>(id);
      int ncells = entries.at(id).dims.at(0); // number of elements is the product of all dimensions
      for (int i=1; i < entries.at(id).dims.size(); i++) {
        ncells *= entries.at(id).dims.at(i);
      }
      // Create an unmanaged yakl::Array that wraps the entry's data pointer and dimensions
      Array<T,1,memSpace,styleC> ret( name.c_str() , (T *) entries.at(id).ptr , ncells );
      // Return the yakl::Array
      return ret;
    }


    // Validate all numerical entries. positive-definite entries are validated to ensure no negative values
    // All floating point values are checked for infinities. All entries are checked for NaNs.
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    // die_on_failed_check : If true, the program will terminate on the first failed check
    //                       If false, the program will continue checking all entries and print warnings
    //                       Default is false
    void validate_all( bool die_on_failed_check = false ) const {
      for (int id = 0; id < entries.size(); id++) { validate( entries.at(id).name , die_on_failed_check ); }
    }


    // Validate one entry. positive-definite entries are validated to ensure no negative values
    // All floating point values are checked for infinities. All entries are checked for NaNs.
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    // name : Name of the entry to validate
    // die_on_failed_check : If true, the program will terminate on the first failed check
    //                       If false, the program will continue checking and print warnings
    //                       Default is false
    // If the entry does not exist, prints an error message and terminates the program
    void validate( std::string name , bool die_on_failed_check = false ) const {
      validate_nan(name,die_on_failed_check);
      validate_inf(name,die_on_failed_check);
      validate_pos(name,die_on_failed_check);
    }


    // Validate one entry for NaNs
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    // name : Name of the entry to validate
    // die_on_failed_check : If true, the program will terminate on the first failed check
    //                       If false, the program will continue checking and print warnings
    //                       Default is false
    // Only works for entries that are either integer or floating point types
    // If the entry does not exist, prints an error message and terminates the program
    void validate_nan( std::string name , bool die_on_failed_check = false ) const {
      bool die = die_on_failed_check;
      int id = find_entry_or_error(name);
      if      (entry_type_is_same<short int>             (id)) { validate_single_nan<short int const>             (name,die); }
      else if (entry_type_is_same<int>                   (id)) { validate_single_nan<int const>                   (name,die); }
      else if (entry_type_is_same<long int>              (id)) { validate_single_nan<long int const>              (name,die); }
      else if (entry_type_is_same<long long int>         (id)) { validate_single_nan<long long int const>         (name,die); }
      else if (entry_type_is_same<unsigned short int>    (id)) { validate_single_nan<unsigned short int const>    (name,die); }
      else if (entry_type_is_same<unsigned int>          (id)) { validate_single_nan<unsigned int const>          (name,die); }
      else if (entry_type_is_same<unsigned long int>     (id)) { validate_single_nan<unsigned long int const>     (name,die); }
      else if (entry_type_is_same<unsigned long long int>(id)) { validate_single_nan<unsigned long long int const>(name,die); }
      else if (entry_type_is_same<float>                 (id)) { validate_single_nan<float const>                 (name,die); }
      else if (entry_type_is_same<double>                (id)) { validate_single_nan<double const>                (name,die); }
      else if (entry_type_is_same<long double>           (id)) { validate_single_nan<long double const>           (name,die); }
    }


    // Validate one entry for infs
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    // name : Name of the entry to validate
    // die_on_failed_check : If true, the program will terminate on the first failed check
    //                       If false, the program will continue checking and print warnings
    //                       Default is false
    // Only works for entries that are floating point types
    // If the entry does not exist, prints an error message and terminates the program
    void validate_inf( std::string name , bool die_on_failed_check = false ) const {
      int id = find_entry_or_error(name);
      if      (entry_type_is_same<float>      (id)) { validate_single_inf<float const>      (name,die_on_failed_check); }
      else if (entry_type_is_same<double>     (id)) { validate_single_inf<double const>     (name,die_on_failed_check); }
      else if (entry_type_is_same<long double>(id)) { validate_single_inf<long double const>(name,die_on_failed_check); }
    }


    // Validate one entry for negative values
    // This is EXPENSIVE. All arrays are copied to the host, and the checks are performed on the host
    // name : Name of the entry to validate
    // die_on_failed_check : If true, the program will terminate on the first failed check
    //                       If false, the program will continue checking and print warnings
    //                       Default is false
    // Only works for entries that are either integer or floating point types
    // If the entry does not exist, prints an error message and terminates the program
    void validate_pos( std::string name , bool die_on_failed_check = false ) const {
      int id = find_entry_or_error(name);
      if      (entry_type_is_same<short int>    (id)) { validate_single_pos<short int const>    (name,die_on_failed_check); }
      else if (entry_type_is_same<int>          (id)) { validate_single_pos<int const>          (name,die_on_failed_check); }
      else if (entry_type_is_same<long int>     (id)) { validate_single_pos<long int const>     (name,die_on_failed_check); }
      else if (entry_type_is_same<long long int>(id)) { validate_single_pos<long long int const>(name,die_on_failed_check); }
      else if (entry_type_is_same<float>        (id)) { validate_single_pos<float const>        (name,die_on_failed_check); }
      else if (entry_type_is_same<double>       (id)) { validate_single_pos<double const>       (name,die_on_failed_check); }
      else if (entry_type_is_same<long double>  (id)) { validate_single_pos<long double const>  (name,die_on_failed_check); }
    }


    // INTERNAL USE: check one entry id for NaNs
    // Validates a single entry for NaNs
    // name : Name of the entry to validate
    // die_on_failed_check : If true, the program will terminate on the first failed check
    //                       If false, the program will continue checking and print warnings
    //                       Default is false
    template <class T>
    void validate_single_nan(std::string name , bool die_on_failed_check = false) const {
      auto arr = get_collapsed<T>(name).createHostCopy();
      for (int i=0; i < arr.get_elem_count(); i++) {
        if ( std::isnan( arr(i) ) ) {
          std::cerr << "WARNING: NaN discovered in: " << name << " at global index: " << i << "\n";
          if (die_on_failed_check) endrun("");
        }
      }
    }


    // INTERNAL USE: check one entry id for infs
    // Validates a single entry for infinities
    // name : Name of the entry to validate
    // die_on_failed_check : If true, the program will terminate on the first failed check
    //                       If false, the program will continue checking and print warnings
    //                       Default is false
    template <class T>
    void validate_single_inf(std::string name , bool die_on_failed_check = false) const {
      auto arr = get_collapsed<T>(name).createHostCopy();
      for (int i=0; i < arr.get_elem_count(); i++) {
        if ( std::isinf( arr(i) ) ) {
          std::cerr << "WARNING: inf discovered in: " << name << " at global index: " << i << "\n";
          if (die_on_failed_check) endrun("");
        }
      }
    }


    // INTERNAL USE: check one entry id for negative values
    // Validates a single entry for negative values
    // name : Name of the entry to validate
    // die_on_failed_check : If true, the program will terminate on the first failed check
    //                       If false, the program will continue checking and print warnings
    //                       Default is false
    template <class T>
    void validate_single_pos(std::string name , bool die_on_failed_check = false) const {
      int id = find_entry_or_error( name );
      if (entries.at(id).positive) {
        auto arr = get_collapsed<T>(name).createHostCopy();
        for (int i=0; i < arr.get_elem_count(); i++) {
          if ( arr(i) < 0. ) {
            std::cerr << "WARNING: negative value discovered in positive-definite entry: " << name
                      << " at global index: " << i << "\n";
            if (die_on_failed_check) endrun("");
          }
        }
      }
    }


    // INTERNAL USE: Return the id of the named entry or -1 if it isn't found
    // Used internally to find entries by name or return -1 if not found
    // name : Name of the entry to find
    // Returns the index of the entry if found, -1 otherwise
    int find_entry( std::string name ) const {
      for (int i=0; i < entries.size(); i++) {
        if (entries.at(i).name == name) return i;
      }
      return -1;
    }


    // INTERNAL USE: Return the id of the named dimension or -1 if it isn't found
    // Used internally to find dimensions by name or return -1 if not found
    // name : Name of the dimension to find
    // Returns the index of the dimension if found, -1 otherwise 
    int find_dimension( std::string name ) const {
      for (int i=0; i < dimensions.size(); i++) {
        if (dimensions.at(i).name == name) return i;
      }
      return -1;
    }


    // INTERNAL USE: Return the id of the named dimension or kill the run if it isn't found
    // Used internally to find dimensions by name or terminate the program if not found
    // name : Name of the dimension to find
    // Returns the index of the dimension if found
    int find_entry_or_error( std::string name ) const {
      int id = find_entry( name );
      if (id >= 0) return id;
      std::cerr << "ERROR: Attempting to retrieve variable name [" << name << "], but it doesn't exist. ";
      endrun("");
      return -1;
    }


    // INTERNAL USE: Return the product of the vector of dimensions
    // Used internally to calculate the total size of an entry based on its dimensions
    // dims : Vector of dimension lengths
    // Returns the product of the dimension lengths
    int get_data_size( std::vector<int> dims ) const {
      int size = 1;
      for (int i=0; i < dims.size(); i++) { size *= dims.at(i); }
      return size;
    }


    // INTERNAL USE: Return the size of the named dimension or kill the run if it isn't found
    // Used internally to get the size of a dimension by name or terminate the program if not found
    // name : Name of the dimension to get the size of
    // Returns the size of the dimension
    int get_dimension_size( std::string name ) const {
      int id = find_dimension( name );
      if (id == -1) {
        std::cerr << "ERROR: Attempting to get size of dimension name [" << name << "], but it doesn't exist. ";
        endrun("ERROR: Could not find dimension.");
      }
      return dimensions.at(id).len;
    }


    // INTERNAL USE: Return the Entry struct for the named entry or kill the run if it isn't found
    // Used internally to get the Entry struct for an entry by name or terminate the program if not found
    // name : Name of the entry to get
    // Returns the Entry struct for the entry
    Entry const & get_entry( std::string name ) const {
      return entries.at(find_entry_or_error(name));
    }


    // INTERNAL USE: Return the C++ hash of this type. Ignore const and volatiles modifiers
    // Used internally to get the type hash for type comparisons to ensure type safety
    // T : Template parameter for the type to get the hash of
    // Returns the hash code of the type T with const and volatile qualifiers removed
    template <class T> size_t get_type_hash() const {
      return typeid(typename std::remove_cv<T>::type).hash_code();
    }


    // INTERNAL USE: Return whether the entry id's type is the same as the templated type
    // Used internally to check if the type of an entry matches the expected type
    // T : Template parameter for the type to compare against the entry's type
    // id : Index of the entry to check
    // Returns true if the types match, false otherwise
    template <class T> size_t entry_type_is_same(int id) const {
      return entries.at(id).type_hash == get_type_hash<T>();
    }


    // INTERNAL USE: Determine if the entry id's type matches the templated type
    // Used internally to validate that the type of an entry matches the expected type
    // T : Template parameter for the type to compare against the entry's type
    // id : Index of the entry to check
    template <class T>
    bool validate_type(int id) const {
      if ( entries.at(id).type_hash != get_type_hash<T>() ) return false;
      return true;
    }


    // INTERNAL USE: End the run if the templated number of dimensions is not the same as the entry id's
    //     number of dimensions
    // Used internally to validate that the number of dimensions of an entry matches the expected number
    // N : Template parameter for the expected number of dimensions
    // id : Index of the entry to check
    // Returns true if the number of dimensions match, false otherwise
    template <int N>
    bool validate_dims(int id) const {
      if ( N != entries.at(id).dims.size() ) return false;
      return true;
    }


    // INTERNAL USE: End the run if the entry id's of dimensions < 2
    // Used internally to validate that the number of dimensions of an entry is at least 2
    // id : Index of the entry to check
    // Returns true if the number of dimensions is at least 2, false otherwise
    bool validate_dims_lev_col(int id) const {
      if ( entries.at(id).dims.size() < 2 ) return false;
      return true;
    }


    // Deallocate all entries, and set the entries and dimensions to empty vectors. This is called by the destructor
    // Generally meant for internal use, but perhaps there are cases where the user might want to call this directly.
    // This method deallocates all entries using the deallocate function pointer
    void finalize() {
      Kokkos::fence();
      for (int i=0; i < entries.size(); i++) {
        deallocate( entries.at(i).ptr , entries.at(i).name.c_str() );
      }
      entries    = std::vector<Entry>();
      dimensions = std::vector<Dimension>();
    }


  };

  
  // Device specialization of the DataManagerTemplate to manage data in device memory
  typedef DataManagerTemplate<yakl::memDevice> DataManager;

  // Host specialization of the DataManagerTemplate to manage data in host memory
  typedef DataManagerTemplate<yakl::memHost> DataManagerHost;

}


