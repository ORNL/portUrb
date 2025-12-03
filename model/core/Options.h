
#pragma once

#include "main_header.h"


namespace core {

  // The Options class manages named scalar options of various types
  // It allows adding, setting, getting, checking existence, deleting, finalizing, and cloning options
  // The class uses type erasure with void pointers and type hashes to store options of arbitrary types
  // The class provides type safety by checking type hashes during get and set operations
  // The class supports move semantics for efficient transfer of ownership of options
  // The class does not support copy semantics to avoid accidental copying of large data structures
  // The class provides detailed error messages and terminates the program on errors
  // Typically, all scalar options in a simulation will be managed by an Options object in the Coupler or other core classes
  class Options {
  public:

    // This struct holds information about a single option
    struct Option {
      std::string key;       // Unique name of the option
      void *      data;      // Pointer to the option data
      size_t      type_hash; // Hash of the data type for type checking for get methods
    };

    std::vector<Option> options; // Vector of all registered options


    // Constructor initializes the Options object
    Options() {}

    // Destructor deallocates all options and clears the options vector
    ~Options() { finalize(); }

    // Move constructor
    Options( Options &&rhs) = default;

    // Move assignment operator
    Options &operator=( Options &&rhs) = default;

    // Delete copy constructor and copy assignment operator to avoid accidental copying of large data structures
    Options( Options const &dm ) = delete;

    // Delete copy assignment operator to avoid accidental copying of large data structures
    Options &operator=( Options const &dm ) = delete;


    // Deallocate all options, and set the options vector to empty. This is called by the destructor
    // Generally meant for internal use, but perhaps there are cases where the user might want to call this directly.
    // This method deallocates all options using the delete_generic function
    void finalize() {
      Kokkos::fence();
      for (int i=0; i < options.size(); i++) {
        delete_generic(i);
      }
      options = std::vector<Option>();
    }


    // Clone this Options into another Options instance
    // rhs : Options instance to clone into
    // The clone_into method allocates new memory for each option in the target Options
    //  and copies the data from this Options into the target instance
    // The method uses the clone_specific method for each option based on its type
    void clone_into( Options & rhs ) const {
      for (int i=0; i < this->options.size(); i++) {
        if      (options.at(i).type_hash == get_type_hash<short int>             ()) { clone_specific<short int>             (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<int>                   ()) { clone_specific<int>                   (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<long int>              ()) { clone_specific<long int>              (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<long long int>         ()) { clone_specific<long long int>         (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<unsigned short int>    ()) { clone_specific<unsigned short int>    (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<unsigned int>          ()) { clone_specific<unsigned int>          (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<unsigned long int>     ()) { clone_specific<unsigned long int>     (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<unsigned long long int>()) { clone_specific<unsigned long long int>(i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<float>                 ()) { clone_specific<float>                 (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<double>                ()) { clone_specific<double>                (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<long double>           ()) { clone_specific<long double>           (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<bool>                  ()) { clone_specific<bool>                  (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<char>                  ()) { clone_specific<char>                  (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<std::string>           ()) { clone_specific<std::string>           (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<std::vector<int>>      ()) { clone_specific<std::vector<int>>      (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<std::vector<float>>    ()) { clone_specific<std::vector<float>>    (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<std::vector<bool>>     ()) { clone_specific<std::vector<bool>>     (i,rhs); }
        else if (options.at(i).type_hash == get_type_hash<std::vector<double>>   ()) { clone_specific<std::vector<double>>   (i,rhs); }
      }
    }


    // Internal use: Clone a specific option of templated type T into the provided Options instance
    // i : Index of the option to clone
    // rhs : Options instance to clone the option into
    // This method allocates new memory for the option in the target Options instance
    //  and copies the data from this Options into the target instance
    // This method is called by the clone_into method for each option based on its type
    template <class T>
    void clone_specific( int i , Options & rhs ) const {
      // Add the option to the target Options instance
      rhs.add_option<T>( options.at(i).key , *static_cast<T *>(options.at(i).data) );
    }


    // Internal use: Delete the option at the given index using the correct type
    // id : Index of the option to delete
    void delete_generic(int id) {
      if      (options.at(id).type_hash == get_type_hash<short int>             ()) { delete_specific<short int>             (id); }
      else if (options.at(id).type_hash == get_type_hash<int>                   ()) { delete_specific<int>                   (id); }
      else if (options.at(id).type_hash == get_type_hash<long int>              ()) { delete_specific<long int>              (id); }
      else if (options.at(id).type_hash == get_type_hash<long long int>         ()) { delete_specific<long long int>         (id); }
      else if (options.at(id).type_hash == get_type_hash<unsigned short int>    ()) { delete_specific<unsigned short int>    (id); }
      else if (options.at(id).type_hash == get_type_hash<unsigned int>          ()) { delete_specific<unsigned int>          (id); }
      else if (options.at(id).type_hash == get_type_hash<unsigned long int>     ()) { delete_specific<unsigned long int>     (id); }
      else if (options.at(id).type_hash == get_type_hash<unsigned long long int>()) { delete_specific<unsigned long long int>(id); }
      else if (options.at(id).type_hash == get_type_hash<float>                 ()) { delete_specific<float>                 (id); }
      else if (options.at(id).type_hash == get_type_hash<double>                ()) { delete_specific<double>                (id); }
      else if (options.at(id).type_hash == get_type_hash<long double>           ()) { delete_specific<long double>           (id); }
      else if (options.at(id).type_hash == get_type_hash<bool>                  ()) { delete_specific<bool>                  (id); }
      else if (options.at(id).type_hash == get_type_hash<char>                  ()) { delete_specific<char>                  (id); }
      else if (options.at(id).type_hash == get_type_hash<std::string>           ()) { delete_specific<std::string>           (id); }
      else if (options.at(id).type_hash == get_type_hash<std::vector<int>>      ()) { delete_specific<std::vector<int>>      (id); }
      else if (options.at(id).type_hash == get_type_hash<std::vector<float>>    ()) { delete_specific<std::vector<float>>    (id); }
      else if (options.at(id).type_hash == get_type_hash<std::vector<bool>>     ()) { delete_specific<std::vector<bool>>     (id); }
      else if (options.at(id).type_hash == get_type_hash<std::vector<double>>   ()) { delete_specific<std::vector<double>>   (id); }
    }


    // Internal use: Delete the option at the given index of templated type T
    // id : Index of the option to delete
    // This method is called by the delete_generic method for each option based on its type
    template <class T>
    void delete_specific(int id) {
      delete (T *) options.at(id).data;
    }


    // Add the option of the templated type
    // T : Template parameter for the type of the option
    // key : Unique name of the option
    // value : Value of the option
    // If an option with the same name already exists, it is overwritten
    // If the key is an empty string, the option is not added
    template <class T>
    void add_option( std::string key , T value ) {
      validate_type<T>();
      if ( key == "" ) return; // Do not add empty keys
      int id = find_option( key ); // Check if the option already exists
      if ( id == -1 ) { // If the option does not exist, add it
        T * ptr = new T(value); // Allocate new memory for the option
        options.push_back({ key , (void *) ptr , get_type_hash<T>() }); // Add the new option to the options vector
      } else { // Otherwise, overwrite the existing option
        *((T *) options.at(id).data) = value;
      }
    }


    // Set the option of the templated type (same as add_option)
    template <class T>
    void set_option( std::string key , T value ) {
      validate_type<T>();
      if ( key == "" ) return; // Do not set empty keys
      add_option( key , value );
    }


    // Get the option of the given name (must match the templated type)
    // T : Template parameter for the type of the option
    // key : Unique name of the option
    // If the option does not exist, prints an error message and terminates the program
    // If the templated type does not match the option's type, prints an error message and terminates the program
    // Returns the value of the option
    template <class T>
    T get_option( std::string key ) const {
      validate_type<T>();
      int id = find_option_or_die( key );
      if (get_type_hash<T>() != options.at(id).type_hash) {
        std::cerr << "ERROR: Requesting option using the wrong type for key [" << key << "]" << std::endl;
        endrun("");
      }
      return *( (T *) options.at(id).data);
    }


    // Internal use: Find the index of the option with the given name
    // key : Unique name of the option to find
    // Returns the index of the option if found, -1 otherwise
    int find_option( std::string key ) const {
      for (int i=0; i < options.size(); i++) {
        if (key == options.at(i).key) return i;
      }
      return -1;
    }


    // Internal use: Find the index of the option with the given name or terminate the program if not found
    // key : Unique name of the option to find
    // Returns the index of the option if found
    int find_option_or_die( std::string key ) const {
      int id = find_option(key);
      if (id >= 0) return id;
      std::cerr << "ERROR: Option not found for key [" << key << "]" << std::endl;
      endrun("");
      return -1;
    }


    // Check if an option with the given name exists
    // key : Unique name of the option to check
    // Returns true if the option exists, false otherwise
    bool option_exists( std::string key ) const {
      return find_option(key) >= 0;
    }


    // Get the number of registered options
    // Returns the number of options currently stored in the Options object
    int get_num_options() const {
      return options.size();
    }


    // Delete the option with the given name
    // key : Unique name of the option to delete
    // If the option does not exist, the method does nothing
    void delete_option( std::string key ) {
      int id = find_option(key);
      if (id >= 0) {
        delete_generic(id);
        options.erase( options.begin() + id );
      }
    }


    // INTERNAL USE: Return the C++ hash of this type. Ignore const and volatiles modifiers
    // Used internally to get the type hash for type comparisons to ensure type safety
    // T : Template parameter for the type to get the hash of
    // Returns the hash code of the type T with const and volatile qualifiers removed
    template <class T> size_t get_type_hash() const {
      return typeid(typename std::remove_cv<T>::type).hash_code();
    }


    // INTERNAL USE: Return whether the templated type is supported by the Options class
    // Used internally to validate that the type of an option is supported
    // T : Template parameter for the type to check
    // Returns true if the type is supported, false otherwise
    template <class T>
    bool type_supported() const {
      if ( get_type_hash<T>() == get_type_hash<short int>             () ||
           get_type_hash<T>() == get_type_hash<int>                   () ||
           get_type_hash<T>() == get_type_hash<long int>              () ||
           get_type_hash<T>() == get_type_hash<long long int>         () ||
           get_type_hash<T>() == get_type_hash<unsigned short int>    () ||
           get_type_hash<T>() == get_type_hash<unsigned int>          () ||
           get_type_hash<T>() == get_type_hash<unsigned long int>     () ||
           get_type_hash<T>() == get_type_hash<unsigned long long int>() ||
           get_type_hash<T>() == get_type_hash<float>                 () ||
           get_type_hash<T>() == get_type_hash<double>                () ||
           get_type_hash<T>() == get_type_hash<long double>           () ||
           get_type_hash<T>() == get_type_hash<bool>                  () ||
           get_type_hash<T>() == get_type_hash<char>                  () ||
           get_type_hash<T>() == get_type_hash<std::string>           () ||
           get_type_hash<T>() == get_type_hash<std::vector<int>>      () ||
           get_type_hash<T>() == get_type_hash<std::vector<float>>    () ||
           get_type_hash<T>() == get_type_hash<std::vector<bool>>     () ||
           get_type_hash<T>() == get_type_hash<std::vector<double>>   () ) return true;
      return false;
    }


    // INTERNAL USE: End the run if the templated type is not supported by the Options class
    // Used internally to validate that the type of an option is supported
    // T : Template parameter for the type to check
    // Terminates the program if the type is not supported
    template <class T>
    void validate_type() const {
      if (! type_supported<T>() ) endrun("ERROR: Options type is not supported");
    }

  };

}


