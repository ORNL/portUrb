#pragma once

#include "main_header.h"
#include "DataManager.h"
#ifdef PORTURB_HAS_PNETCDF
#include "YAKL_pnetcdf.h"
#endif
#include <any>
#include <limits>
#include <memory>
#include <typeindex>
#include <unordered_map>

#ifdef PORTURB_HAS_ADIOS2
#include <adios2.h>
#endif

#if defined(PORTURB_HAS_ADIOS2) == defined(PORTURB_HAS_PNETCDF)
#error "Exactly one portUrb file I/O backend must be enabled"
#endif

namespace core {

  template <class T> struct is_file_io_vector : std::false_type {};
  template <class T, class Allocator> struct is_file_io_vector<std::vector<T,Allocator>> : std::true_type {};

  class FileIO {
    std::string backend;
    int compression_min_bytes;
    bool write_mode = false;
#ifdef PORTURB_HAS_PNETCDF
    std::unique_ptr<yakl::SimplePNetCDF> pnetcdf;
#endif
    std::unordered_map<std::string,size_t> dimensions;
    std::vector<std::any> deferred_views;

    template <class Array>
    void require_owned(Array const &array) const {
      if (array.use_count() == 0) {
        throw std::runtime_error("FileIO requires directly submitted views to have reference-counted ownership");
      }
    }

#ifdef PORTURB_HAS_ADIOS2
    static void validate_compression_parameters(std::string const &compressor, int clevel) {
      bool const compressor_supported = compressor == "blosclz" || compressor == "lz4" ||
                                          compressor == "lz4hc"   || compressor == "zstd";
      if (!compressor_supported) {
        throw std::invalid_argument("ADIOS2 compression compressor must be one of: blosclz, lz4, lz4hc, zstd");
      }
      if (clevel < 1 || clevel > 9) {
        throw std::invalid_argument("ADIOS2 compression clevel must be an integer from 1 through 9");
      }
    }

    struct AdiosVariableDescriptor {
      adios2::Dims shape;
      std::type_index type;
      bool compression_candidate = false;

      AdiosVariableDescriptor(adios2::Dims shape, std::type_index type, bool compression_candidate) :
        shape(std::move(shape)), type(type), compression_candidate(compression_candidate) {}
    };

    struct AdiosState {
      adios2::ADIOS adios;
      adios2::IO io;
      adios2::Engine engine;
      adios2::Operator compression_operator;
      std::unordered_map<std::string,AdiosVariableDescriptor> variables;

      AdiosState(MPI_Comm comm, std::string const &name, std::string const &compressor, int clevel) :
        adios(comm), io(adios.DeclareIO(name)) {
        adios2::Params parameters = {{"compressor",compressor},
                                     {"clevel",std::to_string(clevel)},
                                     {"doshuffle","BLOSC_BITSHUFFLE"}};
        compression_operator = adios.DefineOperator("porturb_blosc2",adios2::ops::LosslessBlosc,parameters);
      }
    };
    std::unique_ptr<AdiosState> adios;
    inline static int adios_instance = 0;
#endif

    template <class Array>
    auto stage_host(Array const &array) {
      require_owned(array);
      if constexpr (Array::on_device) return array.createHostCopy();
      else                            return array;
    }

#ifdef PORTURB_HAS_ADIOS2
    template <class T>
    void define_variable_attribute(T const &value, std::string const &variable_name, std::string const &name) {
      if constexpr (is_file_io_vector<T>::value) {
        using ValueType = typename T::value_type;
        if constexpr (std::is_same_v<ValueType,bool>) {
          std::vector<std::uint8_t> converted(value.begin(),value.end());
          adios->io.DefineAttribute<std::uint8_t>(name,converted.data(),converted.size(),variable_name);
        } else {
          adios->io.DefineAttribute<ValueType>(name,value.data(),value.size(),variable_name);
        }
      } else if constexpr (std::is_same_v<T,bool>) {
        adios->io.DefineAttribute<std::uint8_t>(name,value ? 1 : 0,variable_name);
      } else if constexpr (std::is_same_v<T,char>) {
        adios->io.DefineAttribute<std::int8_t>(name,static_cast<std::int8_t>(value),variable_name);
      } else {
        adios->io.DefineAttribute<T>(name,value,variable_name);
      }
    }

    template <class T>
    adios2::Variable<T> define_variable(std::string const &name) {
      auto &descriptor = adios->variables.at(name);
      if (descriptor.type != std::type_index(typeid(T))) throw std::runtime_error("ADIOS2 variable type mismatch: "+name);
      auto variable = descriptor.shape.empty() ? adios->io.DefineVariable<T>(name) :
                                                 adios->io.DefineVariable<T>(name,descriptor.shape);
      if (descriptor.compression_candidate) variable.AddOperation(adios->compression_operator);
      return variable;
    }

    template <class T>
    adios2::Variable<T> adios_variable(std::string const &name) {
      auto variable = adios->io.InquireVariable<T>(name);
      if (!variable) throw std::runtime_error("ADIOS2 variable type mismatch or variable not found: "+name);
      return variable;
    }

    template <class Array>
    void adios_put(Array const &array, std::string const &name, std::vector<MPI_Offset> const &start,
                   bool contribute = true, bool collective = true) {
      (void) collective;
      using T = typename Array::non_const_value_type;
      auto variable = adios_variable<T>(name);
      if (!contribute) return;
      adios2::Dims selection_start(start.begin(),start.end());
      adios2::Dims selection_count;
      for (int dim = 0; dim < Array::rank(); dim++) selection_count.push_back(array.extent(dim));
      if (std::find(selection_count.begin(),selection_count.end(),0) != selection_count.end()) return;
      auto host = stage_host(array);
      if (!selection_count.empty()) variable.SetSelection({selection_start,selection_count});
      deferred_views.emplace_back(host);
      adios->engine.Put(variable,host.data(),adios2::Mode::Deferred);
    }
#endif

  public:
    static std::string default_backend() {
#ifdef PORTURB_HAS_ADIOS2
      return "adios2";
#else
      return "pnetcdf";
#endif
    }

    static std::string default_compression_compressor() { return "zstd"; }
    static int constexpr default_compression_clevel() { return 5; }

    FileIO(MPI_Comm comm, std::string backend, int compression_min_bytes = 1048576,
           std::string const &compression_compressor = default_compression_compressor(),
           int compression_clevel = default_compression_clevel()) :
      backend(std::move(backend)), compression_min_bytes(compression_min_bytes) {
#ifdef PORTURB_HAS_ADIOS2
      if (this->backend != "adios2") {
        throw std::runtime_error("This portUrb build only supports the adios2 file_io_backend");
      }
      validate_compression_parameters(compression_compressor,compression_clevel);
      adios = std::make_unique<AdiosState>(comm,"porturb_"+std::to_string(adios_instance++),
                                           compression_compressor,compression_clevel);
#else
      (void) compression_compressor;
      (void) compression_clevel;
      if (this->backend != "pnetcdf") {
        throw std::runtime_error("This portUrb build only supports the pnetcdf file_io_backend");
      }
      pnetcdf = std::make_unique<yakl::SimplePNetCDF>(comm);
#endif
    }

    void create(std::string const &filename, int flags = 0, MPI_Info info = MPI_INFO_NULL) {
      write_mode = true;
#ifdef PORTURB_HAS_ADIOS2
      adios->io.SetEngine("BP5");
      adios->engine = adios->io.Open(filename,adios2::Mode::Write);
      adios->engine.BeginStep();
#else
      pnetcdf->create(filename,flags == 0 ? NC_CLOBBER | NC_64BIT_DATA : flags,info);
#endif
    }

    void open(std::string const &filename, int flags = 0) {
      write_mode = false;
#ifdef PORTURB_HAS_ADIOS2
      adios->io.SetEngine("BP5");
      adios->engine = adios->io.Open(filename,adios2::Mode::ReadRandomAccess);
#else
      pnetcdf->open(filename,flags);
#endif
    }

    void close() {
#ifdef PORTURB_HAS_ADIOS2
      if (adios->engine) {
        if (write_mode) adios->engine.EndStep();
        deferred_views.clear();
        adios->engine.Close();
      }
#else
      pnetcdf->close();
#endif
    }

    void create_dim(std::string const &name, size_t extent) {
      dimensions[name] = extent;
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->create_dim(name,extent);
#endif
    }

    bool dim_exists(std::string const &name) const {
#ifdef PORTURB_HAS_ADIOS2
      return dimensions.find(name) != dimensions.end();
#else
      return pnetcdf->dim_exists(name);
#endif
    }

    size_t get_dim_size(std::string const &name) const {
#ifdef PORTURB_HAS_ADIOS2
      auto const known = dimensions.find(name);
      if (known != dimensions.end()) return known->second;
      auto variable = adios->io.InquireVariable(name);
      if (!variable || variable.Shape().size() != 1) {
        throw std::runtime_error("ADIOS2 one-dimensional coordinate variable not found: "+name);
      }
      return variable.Shape().at(0);
#else
      return pnetcdf->get_dim_size(name);
#endif
    }

    bool var_exists(std::string const &name) const {
#ifdef PORTURB_HAS_ADIOS2
      if (write_mode && adios->variables.find(name) != adios->variables.end()) return true;
      return adios->io.VariableType(name) != "";
#else
      return pnetcdf->var_exists(name);
#endif
    }

    bool variable_has_operations(std::string const &name) const {
#ifdef PORTURB_HAS_ADIOS2
      auto variable = adios->io.InquireVariable(name);
      if (!variable) throw std::runtime_error("ADIOS2 variable not found: "+name);
      return !variable.Operations().empty();
#else
      (void) name;
      return false;
#endif
    }

    std::string compression_parameter(std::string const &parameter) const {
#ifdef PORTURB_HAS_ADIOS2
      auto const &parameters = adios->compression_operator.Parameters();
      auto const found = parameters.find(parameter);
      if (found == parameters.end()) throw std::runtime_error("ADIOS2 compression parameter not found: "+parameter);
      return found->second;
#else
      (void) parameter;
      return "";
#endif
    }

    template <class T>
    void create_var(std::string const &name, std::vector<std::string> const &dimnames) {
#ifdef PORTURB_HAS_ADIOS2
      adios2::Dims shape;
      for (auto const &dimname : dimnames) shape.push_back(dimensions.at(dimname));
      size_t total_bytes = sizeof(T);
      for (auto const extent : shape) {
        if (extent != 0 && total_bytes > std::numeric_limits<size_t>::max()/extent) {
          throw std::runtime_error("ADIOS2 variable size overflows size_t: "+name);
        }
        total_bytes *= extent;
      }
      bool const compression_candidate = !shape.empty() && total_bytes >= static_cast<size_t>(compression_min_bytes);
      auto const inserted = adios->variables.emplace(std::piecewise_construct,
                                                     std::forward_as_tuple(name),
                                                     std::forward_as_tuple(shape,std::type_index(typeid(T)),
                                                                           compression_candidate));
      if (!inserted.second) throw std::runtime_error("Duplicate ADIOS2 variable definition: "+name);
      define_variable<T>(name);
#else
      pnetcdf->create_var<T>(name,dimnames);
#endif
    }

    void redef() {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->redef();
#endif
    }
    void enddef() {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->enddef();
#endif
    }
    void begin_indep_data() {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->begin_indep_data();
#endif
    }
    void end_indep_data() {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->end_indep_data();
#endif
    }

    template <class T>
    void writeGlobalAttribute(T const &value, std::string const &name) {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->writeGlobalAttribute(value,name);
#else
        if constexpr (is_file_io_vector<T>::value) {
          using ValueType = typename T::value_type;
          if constexpr (std::is_same_v<ValueType,bool>) {
            std::vector<std::uint8_t> converted(value.begin(),value.end());
            adios->io.DefineAttribute<std::uint8_t>(name,converted.data(),converted.size());
          } else {
            adios->io.DefineAttribute<ValueType>(name,value.data(),value.size());
          }
        } else if constexpr (std::is_same_v<T,bool>) {
          adios->io.DefineAttribute<std::uint8_t>(name,value ? 1 : 0);
        } else if constexpr (std::is_same_v<T,char>) {
          adios->io.DefineAttribute<std::int8_t>(name,static_cast<std::int8_t>(value));
        } else {
          adios->io.DefineAttribute<T>(name,value);
        }
#endif
    }

    template <class T>
    void writeVariableAttribute(T const &value, std::string const &variable_name, std::string const &name) {
#ifdef PORTURB_HAS_PNETCDF
      if (name != "coordinates") pnetcdf->writeVariableAttribute(value,variable_name,name);
#else
      define_variable_attribute(value,variable_name,name);
#endif
    }

    template <class T>
    void readGlobalAttribute(T &value, std::string const &name) const {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->readGlobalAttribute(value,name);
#else
        if constexpr (is_file_io_vector<T>::value) {
          using ValueType = typename T::value_type;
          if constexpr (std::is_same_v<ValueType,bool>) {
            auto attribute = adios->io.InquireAttribute<std::uint8_t>(name);
            if (!attribute) throw std::runtime_error("ADIOS2 global attribute not found: "+name);
            auto data = attribute.Data();
            value.assign(data.size(),false);
            for (size_t i = 0; i < data.size(); i++) value[i] = data[i] != 0;
          } else {
            auto attribute = adios->io.InquireAttribute<ValueType>(name);
            if (!attribute) throw std::runtime_error("ADIOS2 global attribute not found: "+name);
            auto data = attribute.Data();
            value.assign(data.begin(),data.end());
          }
        } else if constexpr (std::is_same_v<T,bool>) {
          auto attribute = adios->io.InquireAttribute<std::uint8_t>(name);
          if (!attribute) throw std::runtime_error("ADIOS2 global attribute not found: "+name);
          value = attribute.Data().at(0) != 0;
        } else if constexpr (std::is_same_v<T,char>) {
          auto attribute = adios->io.InquireAttribute<std::int8_t>(name);
          if (!attribute) throw std::runtime_error("ADIOS2 global attribute not found: "+name);
          value = static_cast<char>(attribute.Data().at(0));
        } else {
          auto attribute = adios->io.InquireAttribute<T>(name);
          if (!attribute) throw std::runtime_error("ADIOS2 global attribute not found: "+name);
          value = attribute.Data().at(0);
        }
#endif
    }

    template <class T>
    void readVariableAttribute(T &value, std::string const &variable_name, std::string const &name) const {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->readVariableAttribute(value,variable_name,name);
#else
      if constexpr (is_file_io_vector<T>::value) {
        using ValueType = typename T::value_type;
        if constexpr (std::is_same_v<ValueType,bool>) {
          auto attribute = adios->io.InquireAttribute<std::uint8_t>(name,variable_name);
          if (!attribute) throw std::runtime_error("ADIOS2 variable attribute not found: "+variable_name+"/"+name);
          auto data = attribute.Data();
          value.assign(data.size(),false);
          for (size_t i = 0; i < data.size(); i++) value[i] = data[i] != 0;
        } else {
          auto attribute = adios->io.InquireAttribute<ValueType>(name,variable_name);
          if (!attribute) throw std::runtime_error("ADIOS2 variable attribute not found: "+variable_name+"/"+name);
          auto data = attribute.Data();
          value.assign(data.begin(),data.end());
        }
      } else if constexpr (std::is_same_v<T,bool>) {
        auto attribute = adios->io.InquireAttribute<std::uint8_t>(name,variable_name);
        if (!attribute) throw std::runtime_error("ADIOS2 variable attribute not found: "+variable_name+"/"+name);
        value = attribute.Data().at(0) != 0;
      } else if constexpr (std::is_same_v<T,char>) {
        auto attribute = adios->io.InquireAttribute<std::int8_t>(name,variable_name);
        if (!attribute) throw std::runtime_error("ADIOS2 variable attribute not found: "+variable_name+"/"+name);
        value = static_cast<char>(attribute.Data().at(0));
      } else {
        auto attribute = adios->io.InquireAttribute<T>(name,variable_name);
        if (!attribute) throw std::runtime_error("ADIOS2 variable attribute not found: "+variable_name+"/"+name);
        value = attribute.Data().at(0);
      }
#endif
    }

    template <class Array>
    void write_all(Array const &array, std::string const &name, std::vector<MPI_Offset> const &start,
                   bool adios_writer = true) {
      require_owned(array);
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->write_all(array,name,start);
#else
      adios_put(array,name,start,adios_writer,true);
#endif
    }

    // DataManager views are intentionally unmanaged. Convert them to an owning view before entering the
    // direct-view path, where deferred ADIOS2 writes require reference-counted ownership.
    template <class SourceType, int Rank, class FileType = SourceType>
    void write_data_manager(DataManager<> const &dm, std::string const &entry_name, std::string const &file_name,
                            std::vector<MPI_Offset> const &start) {
      auto owned = dm.template get<SourceType const,Rank>(entry_name).template as<FileType>();
      write_all(owned,file_name,start);
    }

    template <class SourceType, int Rank, class FileType = SourceType>
    void write_data_manager(DataManager<> const &dm, std::string const &entry_name, std::string const &file_name) {
      auto owned = dm.template get<SourceType const,Rank>(entry_name).template as<FileType>();
      write(owned,file_name);
    }

    template <class Array>
    void write(Array const &array, std::string const &name) {
      require_owned(array);
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->write(array,name);
#else
      adios_put(array,name,std::vector<MPI_Offset>(Array::rank(),0),true,false);
#endif
    }

    template <class T> requires std::is_arithmetic_v<T>
    void write(T const &value, std::string const &name) {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->write(value,name);
#else
      auto stored = std::make_shared<T>(value);
      auto variable = adios_variable<T>(name);
      deferred_views.emplace_back(stored);
      adios->engine.Put(variable,stored.get(),adios2::Mode::Deferred);
#endif
    }

    template <class Array>
    void read_all(Array const &array, std::string const &name, std::vector<MPI_Offset> const &start) {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->read_all(array,name,start);
#else
      using T = typename Array::non_const_value_type;
      auto host = array.createHostObject();
      auto variable = adios_variable<T>(name);
      adios2::Dims selection_start(start.begin(),start.end());
      adios2::Dims selection_count;
      for (int dim = 0; dim < Array::rank(); dim++) selection_count.push_back(array.extent(dim));
      if (std::find(selection_count.begin(),selection_count.end(),0) != selection_count.end()) return;
      variable.SetSelection({selection_start,selection_count});
      variable.SetStepSelection({0,1});
      adios->engine.Get(variable,host.data(),adios2::Mode::Sync);
      host.deep_copy_to(array);
#endif
    }

    template <class T> requires std::is_arithmetic_v<T>
    void read(T &value, std::string const &name) {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->read(value,name);
#else
      auto variable = adios_variable<T>(name);
      variable.SetStepSelection({0,1});
      adios->engine.Get(variable,&value,adios2::Mode::Sync);
#endif
    }

    template <class Array>
    void read(Array const &array, std::string const &name) {
#ifdef PORTURB_HAS_PNETCDF
      pnetcdf->read(array,name);
#else
      read_all(array,name,std::vector<MPI_Offset>(Array::rank(),0));
#endif
    }
  };

}
