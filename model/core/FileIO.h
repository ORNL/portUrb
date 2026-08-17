#pragma once

#include "main_header.h"
#include "DataManager.h"
#ifdef PORTURB_HAS_PNETCDF
#include "YAKL_pnetcdf.h"
#endif
#include <any>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

#ifdef PORTURB_HAS_ADIOS2
#include <adios2.h>
#include <blosc2.h>
#endif

#if defined(PORTURB_HAS_ADIOS2) == defined(PORTURB_HAS_PNETCDF)
#error "Exactly one portUrb file I/O backend must be enabled"
#endif

namespace core {

  template <class T> struct is_file_io_vector : std::false_type {};
  template <class T, class Allocator> struct is_file_io_vector<std::vector<T,Allocator>> : std::true_type {};

  class FileIO {
    std::string backend;
    MPI_Comm comm;
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
    struct AdiosVariableDescriptor {
      adios2::Dims shape;
      std::type_index type;
      bool compression_candidate = false;
      bool defined = false;
      bool compressed = false;
      bool written = false;
      std::vector<std::function<void()>> attributes;

      AdiosVariableDescriptor(adios2::Dims shape, std::type_index type, bool compression_candidate) :
        shape(std::move(shape)), type(type), compression_candidate(compression_candidate) {}
    };

    struct AdiosState {
      adios2::ADIOS adios;
      adios2::IO io;
      adios2::Engine engine;
      std::unordered_map<std::string,AdiosVariableDescriptor> variables;
      bool codec_schema_defined = false;

      AdiosState(MPI_Comm comm, std::string const &name) : adios(comm), io(adios.DeclareIO(name)) {}
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
    static size_t checked_product(adios2::Dims const &dims) {
      size_t result = 1;
      for (auto const extent : dims) {
        if (extent == 0 || result > std::numeric_limits<size_t>::max()/extent) {
          throw std::runtime_error("Invalid or overflowing BP5 codec shape");
        }
        result *= extent;
      }
      return result;
    }

    static void append_u64(std::vector<std::uint8_t> &buffer, std::uint64_t value) {
      for (int byte = 0; byte < 8; byte++) buffer.push_back(static_cast<std::uint8_t>(value >> (8*byte)));
    }

    static std::uint64_t read_u64(std::vector<std::uint8_t> const &buffer, size_t &offset) {
      if (offset > buffer.size() || buffer.size()-offset < 8) {
        throw std::runtime_error("Truncated BP5 codec block header");
      }
      std::uint64_t value = 0;
      for (int byte = 0; byte < 8; byte++) value |= static_cast<std::uint64_t>(buffer[offset+byte]) << (8*byte);
      offset += 8;
      return value;
    }

    static std::string shape_string(adios2::Dims const &shape) {
      std::ostringstream stream;
      for (size_t dim = 0; dim < shape.size(); dim++) {
        if (dim != 0) stream << ",";
        stream << shape[dim];
      }
      return stream.str();
    }

    static adios2::Dims parse_shape(std::string const &text) {
      adios2::Dims shape;
      std::istringstream stream(text);
      std::string token;
      while (std::getline(stream,token,',')) {
        if (token.empty()) throw std::runtime_error("Invalid empty dimension in BP5 codec_shape");
        size_t consumed = 0;
        auto const extent = std::stoull(token,&consumed);
        if (consumed != token.size() || extent == 0) throw std::runtime_error("Invalid BP5 codec_shape: "+text);
        shape.push_back(extent);
      }
      if (shape.empty()) throw std::runtime_error("Empty BP5 codec_shape");
      return shape;
    }

    template <class T>
    static std::string codec_dtype() {
      if constexpr (std::is_same_v<T,float>)               return "<f4";
      else if constexpr (std::is_same_v<T,double>)         return "<f8";
      else if constexpr (std::is_same_v<T,std::int32_t>)   return "<i4";
      else if constexpr (std::is_same_v<T,std::int64_t>)   return "<i8";
      else if constexpr (std::is_same_v<T,std::uint8_t> || std::is_same_v<T,unsigned char>) return "|u1";
      else if constexpr (std::is_same_v<T,std::int8_t> || std::is_same_v<T,signed char>)     return "|i1";
      else return "";
    }

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

    void define_pending_attributes(std::string const &name) {
      auto &descriptor = adios->variables.at(name);
      for (auto &define : descriptor.attributes) define();
      descriptor.attributes.clear();
    }

    template <class T>
    adios2::Variable<T> define_ordinary_variable(std::string const &name) {
      auto &descriptor = adios->variables.at(name);
      if (descriptor.type != std::type_index(typeid(T))) throw std::runtime_error("ADIOS2 variable type mismatch: "+name);
      auto variable = descriptor.shape.empty() ? adios->io.DefineVariable<T>(name) :
                                                 adios->io.DefineVariable<T>(name,descriptor.shape);
      descriptor.defined = true;
      define_pending_attributes(name);
      return variable;
    }

    template <class T>
    adios2::Variable<T> adios_variable(std::string const &name) {
      auto variable = adios->io.InquireVariable<T>(name);
      if (!variable) throw std::runtime_error("ADIOS2 variable type mismatch or variable not found: "+name);
      return variable;
    }

    static void mpi_check(int code, std::string const &operation) {
      if (code != MPI_SUCCESS) throw std::runtime_error("MPI failure during "+operation);
    }

    template <class Array>
    void adios_put_compressed(Array const &array, std::string const &name, std::vector<MPI_Offset> const &start,
                              bool contribute) {
      using T = typename Array::non_const_value_type;
      auto host = stage_host(array);
      auto &descriptor = adios->variables.at(name);
      adios2::Dims block_start(start.begin(),start.end());
      adios2::Dims block_count;
      for (int dim = 0; dim < Array::rank(); dim++) block_count.push_back(array.extent(dim));
      if (block_start.size() != descriptor.shape.size() || block_count.size() != descriptor.shape.size()) {
        throw std::runtime_error("BP5 codec selection rank mismatch for variable: "+name);
      }
      if (std::find(block_count.begin(),block_count.end(),0) != block_count.end()) contribute = false;

      std::vector<std::vector<std::uint8_t>> compressed_chunks;
      std::string compression_error;
      if (contribute) {
        try {
          size_t const input_bytes = host.size()*sizeof(T);
          size_t constexpr target_chunk_bytes = size_t(1) << 30;
          size_t chunk_bytes = std::min(target_chunk_bytes,static_cast<size_t>(BLOSC2_MAX_BUFFERSIZE));
          chunk_bytes -= chunk_bytes % sizeof(T);
          if (chunk_bytes == 0) throw std::runtime_error("BP5 codec chunk size is smaller than one element");

          auto params = BLOSC2_CPARAMS_DEFAULTS;
          params.compcode = BLOSC_LZ4;
          params.clevel = 5;
          params.typesize = static_cast<int32_t>(sizeof(T));
          params.filters[BLOSC2_MAX_FILTERS-1] = BLOSC_BITSHUFFLE;
          std::unique_ptr<blosc2_context,decltype(&blosc2_free_ctx)> context(blosc2_create_cctx(params),blosc2_free_ctx);
          if (!context) throw std::runtime_error("could not create the Blosc2 LZ4 context");
          for (size_t offset = 0; offset < input_bytes; offset += chunk_bytes) {
            size_t const source_bytes = std::min(chunk_bytes,input_bytes-offset);
            std::vector<std::uint8_t> compressed(source_bytes+BLOSC2_MAX_OVERHEAD);
            int const result = blosc2_compress_ctx(context.get(),
                                                   reinterpret_cast<std::uint8_t const *>(host.data())+offset,
                                                   static_cast<int32_t>(source_bytes),compressed.data(),
                                                   static_cast<int32_t>(compressed.size()));
            if (result <= 0) {
              throw std::runtime_error("Blosc2 LZ4 compression failed with code "+std::to_string(result));
            }
            compressed.resize(result);
            compressed_chunks.push_back(std::move(compressed));
          }
        } catch (std::exception const &error) {
          compression_error = error.what();
        }
      }

      int local_success = compression_error.empty() ? 1 : 0;
      int global_success = 0;
      mpi_check(MPI_Allreduce(&local_success,&global_success,1,MPI_INT,MPI_MIN,comm),"BP5 compression agreement");
      if (global_success == 0) {
        throw std::runtime_error("Collective BP5 compression failure for "+name+
                                 (compression_error.empty() ? " on another rank" : ": "+compression_error));
      }

      int writer_id = 0;
      mpi_check(MPI_Comm_rank(comm,&writer_id),"BP5 writer rank lookup");
      auto payload = std::make_shared<std::vector<std::uint8_t>>();
      if (contribute) {
        payload->insert(payload->end(),{'B','P','5','C','O','D','C',1});
        append_u64(*payload,writer_id);
        append_u64(*payload,block_start.size());
        for (auto const value : block_start) append_u64(*payload,value);
        for (auto const value : block_count) append_u64(*payload,value);
        append_u64(*payload,compressed_chunks.size());
        for (auto const &chunk : compressed_chunks) append_u64(*payload,chunk.size());
        for (auto const &chunk : compressed_chunks) payload->insert(payload->end(),chunk.begin(),chunk.end());
      }

      std::uint64_t local_bytes = payload->size();
      std::uint64_t total_bytes = 0;
      std::uint64_t byte_offset = 0;
      std::uint64_t local_blocks = contribute ? 1 : 0;
      std::uint64_t total_blocks = 0;
      std::uint64_t row_offset = 0;
      mpi_check(MPI_Exscan(&local_bytes,&byte_offset,1,MPI_UINT64_T,MPI_SUM,comm),"BP5 byte offset scan");
      mpi_check(MPI_Allreduce(&local_bytes,&total_bytes,1,MPI_UINT64_T,MPI_SUM,comm),"BP5 byte total reduction");
      mpi_check(MPI_Exscan(&local_blocks,&row_offset,1,MPI_UINT64_T,MPI_SUM,comm),"BP5 block row scan");
      mpi_check(MPI_Allreduce(&local_blocks,&total_blocks,1,MPI_UINT64_T,MPI_SUM,comm),"BP5 block count reduction");
      if (writer_id == 0) {
        byte_offset = 0;
        row_offset = 0;
      }
      if (total_blocks == 0 || total_bytes == 0) throw std::runtime_error("BP5 compressed variable has no blocks: "+name);

      auto variable = adios->io.DefineVariable<std::uint8_t>(name,{total_bytes});
      size_t const directory_columns = 3+2*descriptor.shape.size();
      auto directory = adios->io.DefineVariable<std::uint64_t>(name+"/codec_block_directory",
                                                               {total_blocks,directory_columns});
      descriptor.defined = true;
      descriptor.compressed = true;
      descriptor.written = true;
      define_pending_attributes(name);
      if (!adios->codec_schema_defined) {
        adios->io.DefineAttribute<std::string>("codec_schema","bp5-codec-v1");
        adios->codec_schema_defined = true;
      }
      define_variable_attribute(std::string("blosc2"),name,"codec");
      define_variable_attribute(codec_dtype<T>(),name,"codec_dtype");
      define_variable_attribute(shape_string(descriptor.shape),name,"codec_shape");
      define_variable_attribute(std::to_string(total_blocks),name,"codec_nblocks");
      define_variable_attribute(std::string("lz4"),name,"codec_compressor");
      define_variable_attribute(std::string("5"),name,"codec_clevel");
      define_variable_attribute(std::string("BLOSC_BITSHUFFLE"),name,"codec_doshuffle");

      if (contribute) {
        variable.SetSelection({{byte_offset},{local_bytes}});
        auto directory_row = std::make_shared<std::vector<std::uint64_t>>(directory_columns);
        (*directory_row)[0] = byte_offset;
        (*directory_row)[1] = local_bytes;
        (*directory_row)[2] = writer_id;
        for (size_t dim = 0; dim < block_start.size(); dim++) {
          (*directory_row)[3+dim] = block_start[dim];
          (*directory_row)[3+block_start.size()+dim] = block_count[dim];
        }
        directory.SetSelection({{row_offset,0},{1,directory_columns}});
        deferred_views.emplace_back(payload);
        deferred_views.emplace_back(directory_row);
        adios->engine.Put(variable,payload->data(),adios2::Mode::Deferred);
        adios->engine.Put(directory,directory_row->data(),adios2::Mode::Deferred);
      }
    }

    template <class Array>
    void adios_put(Array const &array, std::string const &name, std::vector<MPI_Offset> const &start,
                   bool contribute = true, bool collective = true) {
      using T = typename Array::non_const_value_type;
      auto &descriptor = adios->variables.at(name);
      if (!descriptor.defined && descriptor.compression_candidate && collective) {
        adios_put_compressed(array,name,start,contribute);
        return;
      }
      auto variable = descriptor.defined ? adios_variable<T>(name) : define_ordinary_variable<T>(name);
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

    static bool boxes_overlap(std::uint64_t const *start_a, std::uint64_t const *count_a,
                              std::uint64_t const *start_b, std::uint64_t const *count_b, size_t ndim) {
      for (size_t dim = 0; dim < ndim; dim++) {
        if (start_a[dim]+count_a[dim] <= start_b[dim] || start_b[dim]+count_b[dim] <= start_a[dim]) return false;
      }
      return true;
    }

    template <class Array>
    void adios_read_compressed(Array const &array, std::string const &name, std::vector<MPI_Offset> const &start) {
      using T = typename Array::non_const_value_type;
      auto codec_attribute = adios->io.InquireAttribute<std::string>("codec",name);
      auto dtype_attribute = adios->io.InquireAttribute<std::string>("codec_dtype",name);
      auto shape_attribute = adios->io.InquireAttribute<std::string>("codec_shape",name);
      auto blocks_attribute = adios->io.InquireAttribute<std::string>("codec_nblocks",name);
      auto schema_attribute = adios->io.InquireAttribute<std::string>("codec_schema");
      if (!codec_attribute || !dtype_attribute || !shape_attribute || !blocks_attribute || !schema_attribute) {
        throw std::runtime_error("Incomplete BP5 codec metadata for variable: "+name);
      }
      if (schema_attribute.Data().at(0) != "bp5-codec-v1") throw std::runtime_error("Unsupported BP5 codec schema");
      if (codec_attribute.Data().at(0) != "blosc2") throw std::runtime_error("Unsupported BP5 codec for "+name);
      if (dtype_attribute.Data().at(0) != codec_dtype<T>()) throw std::runtime_error("BP5 codec dtype mismatch for "+name);
      auto const global_shape = parse_shape(shape_attribute.Data().at(0));
      if (global_shape.size() != Array::rank() || start.size() != global_shape.size()) {
        throw std::runtime_error("BP5 codec read rank mismatch for variable: "+name);
      }
      checked_product(global_shape);
      adios2::Dims target_start(start.begin(),start.end());
      adios2::Dims target_count;
      for (int dim = 0; dim < Array::rank(); dim++) {
        target_count.push_back(array.extent(dim));
        if (target_start[dim] > global_shape[dim] || target_count[dim] > global_shape[dim]-target_start[dim]) {
          throw std::runtime_error("BP5 codec read selection is out of bounds for variable: "+name);
        }
      }

      auto payload_variable = adios_variable<std::uint8_t>(name);
      auto directory_variable = adios_variable<std::uint64_t>(name+"/codec_block_directory");
      auto const payload_shape = payload_variable.Shape();
      auto const directory_shape = directory_variable.Shape();
      size_t const ndim = global_shape.size();
      if (payload_shape.size() != 1 || directory_shape.size() != 2 || directory_shape[0] == 0 ||
          directory_shape[1] != 3+2*ndim) {
        throw std::runtime_error("Invalid BP5 codec payload or directory shape for variable: "+name);
      }
      size_t consumed = 0;
      auto const declared_blocks = std::stoull(blocks_attribute.Data().at(0),&consumed);
      if (consumed != blocks_attribute.Data().at(0).size() || declared_blocks != directory_shape[0]) {
        throw std::runtime_error("BP5 codec_nblocks does not match the directory for variable: "+name);
      }

      size_t const columns = directory_shape[1];
      std::vector<std::uint64_t> directory(directory_shape[0]*columns);
      directory_variable.SetSelection({{0,0},directory_shape});
      directory_variable.SetStepSelection({0,1});
      adios->engine.Get(directory_variable,directory.data(),adios2::Mode::Sync);

      size_t logical_volume = 0;
      size_t physical_bytes = 0;
      for (size_t block = 0; block < directory_shape[0]; block++) {
        auto const *row = directory.data()+block*columns;
        if (row[1] == 0 || row[0] > payload_shape[0] || row[1] > payload_shape[0]-row[0]) {
          throw std::runtime_error("BP5 codec directory byte range is invalid for variable: "+name);
        }
        size_t block_volume = 1;
        for (size_t dim = 0; dim < ndim; dim++) {
          auto const block_start = row[3+dim];
          auto const block_count = row[3+ndim+dim];
          if (block_count == 0 || block_start > global_shape[dim] || block_count > global_shape[dim]-block_start ||
              block_volume > std::numeric_limits<size_t>::max()/block_count) {
            throw std::runtime_error("BP5 codec directory logical range is invalid for variable: "+name);
          }
          block_volume *= block_count;
        }
        if (logical_volume > std::numeric_limits<size_t>::max()-block_volume ||
            physical_bytes > std::numeric_limits<size_t>::max()-row[1]) {
          throw std::runtime_error("BP5 codec directory coverage overflows size_t for variable: "+name);
        }
        logical_volume += block_volume;
        physical_bytes += row[1];
        for (size_t other = 0; other < block; other++) {
          auto const *other_row = directory.data()+other*columns;
          bool const byte_overlap = row[0] < other_row[0]+other_row[1] && other_row[0] < row[0]+row[1];
          if (byte_overlap || boxes_overlap(row+3,row+3+ndim,other_row+3,other_row+3+ndim,ndim)) {
            throw std::runtime_error("Overlapping BP5 codec directory blocks for variable: "+name);
          }
        }
      }
      if (logical_volume != checked_product(global_shape) || physical_bytes != payload_shape[0]) {
        throw std::runtime_error("BP5 codec directory does not completely tile variable: "+name);
      }

      auto host = array.createHostObject();
      for (size_t block = 0; block < directory_shape[0]; block++) {
        auto const *row = directory.data()+block*columns;
        adios2::Dims intersection_start(ndim);
        adios2::Dims intersection_count(ndim);
        bool intersects = true;
        for (size_t dim = 0; dim < ndim; dim++) {
          intersection_start[dim] = std::max<std::uint64_t>(target_start[dim],row[3+dim]);
          auto const intersection_end = std::min<std::uint64_t>(target_start[dim]+target_count[dim],
                                                                 row[3+dim]+row[3+ndim+dim]);
          if (intersection_end <= intersection_start[dim]) intersects = false;
          intersection_count[dim] = intersects ? intersection_end-intersection_start[dim] : 0;
        }
        if (!intersects) continue;

        std::vector<std::uint8_t> payload(row[1]);
        payload_variable.SetSelection({{row[0]},{row[1]}});
        payload_variable.SetStepSelection({0,1});
        adios->engine.Get(payload_variable,payload.data(),adios2::Mode::Sync);
        std::uint8_t constexpr magic[8] = {'B','P','5','C','O','D','C',1};
        if (payload.size() < 8 || std::memcmp(payload.data(),magic,8) != 0) {
          throw std::runtime_error("Invalid BP5 codec block magic for variable: "+name);
        }
        size_t offset = 8;
        auto const writer_id = read_u64(payload,offset);
        auto const header_ndim = read_u64(payload,offset);
        if (header_ndim != ndim) throw std::runtime_error("BP5 codec block ndim mismatch for variable: "+name);
        adios2::Dims block_start(ndim);
        adios2::Dims block_count(ndim);
        for (size_t dim = 0; dim < ndim; dim++) block_start[dim] = read_u64(payload,offset);
        for (size_t dim = 0; dim < ndim; dim++) block_count[dim] = read_u64(payload,offset);
        if (writer_id != row[2]) throw std::runtime_error("BP5 codec block writer_id mismatch for variable: "+name);
        for (size_t dim = 0; dim < ndim; dim++) {
          if (block_start[dim] != row[3+dim] || block_count[dim] != row[3+ndim+dim]) {
            throw std::runtime_error("BP5 codec block placement mismatch for variable: "+name);
          }
        }
        auto const num_chunks = read_u64(payload,offset);
        if (num_chunks == 0 || num_chunks > (payload.size()-offset)/8) {
          throw std::runtime_error("Invalid BP5 codec chunk count for variable: "+name);
        }
        std::vector<std::uint64_t> chunk_sizes(num_chunks);
        size_t total_compressed = 0;
        for (size_t chunk = 0; chunk < num_chunks; chunk++) {
          chunk_sizes[chunk] = read_u64(payload,offset);
          if (chunk_sizes[chunk] == 0 || total_compressed > std::numeric_limits<size_t>::max()-chunk_sizes[chunk]) {
            throw std::runtime_error("Invalid BP5 codec chunk size for variable: "+name);
          }
          total_compressed += chunk_sizes[chunk];
        }
        if (total_compressed != payload.size()-offset) {
          throw std::runtime_error("BP5 codec chunk sizes do not match payload for variable: "+name);
        }

        size_t const block_elements = checked_product(block_count);
        if (block_elements > std::numeric_limits<size_t>::max()/sizeof(T)) {
          throw std::runtime_error("BP5 codec decoded size overflows size_t for variable: "+name);
        }
        size_t const expected_bytes = block_elements*sizeof(T);
        std::vector<std::uint8_t> decoded;
        decoded.reserve(expected_bytes);
        auto params = BLOSC2_DPARAMS_DEFAULTS;
        std::unique_ptr<blosc2_context,decltype(&blosc2_free_ctx)> context(blosc2_create_dctx(params),blosc2_free_ctx);
        if (!context) throw std::runtime_error("Could not create the Blosc2 decompression context");
        for (auto const chunk_size : chunk_sizes) {
          if (chunk_size > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("Blosc2 frame exceeds the v1 chunk-size limit for variable: "+name);
          }
          int32_t uncompressed_size = 0;
          int32_t compressed_size = 0;
          int32_t block_size = 0;
          int const info = blosc2_cbuffer_sizes(payload.data()+offset,&uncompressed_size,&compressed_size,&block_size);
          if (info < 0 || compressed_size != chunk_size || uncompressed_size <= 0) {
            throw std::runtime_error("Invalid Blosc2 frame metadata for variable: "+name);
          }
          size_t const decoded_offset = decoded.size();
          decoded.resize(decoded_offset+uncompressed_size);
          int const result = blosc2_decompress_ctx(context.get(),payload.data()+offset,chunk_size,
                                                   decoded.data()+decoded_offset,uncompressed_size);
          if (result != uncompressed_size) {
            throw std::runtime_error("Blosc2 decompression failed for variable: "+name);
          }
          offset += chunk_size;
        }
        if (decoded.size() != expected_bytes) {
          throw std::runtime_error("BP5 codec decoded byte count mismatch for variable: "+name);
        }

        adios2::Dims block_strides(ndim,1);
        adios2::Dims target_strides(ndim,1);
        for (size_t dim = ndim-1; dim > 0; dim--) {
          block_strides[dim-1] = block_strides[dim]*block_count[dim];
          target_strides[dim-1] = target_strides[dim]*target_count[dim];
        }
        size_t const intersection_volume = checked_product(intersection_count);
        for (size_t element = 0; element < intersection_volume; element++) {
          size_t remainder = element;
          size_t block_index = 0;
          size_t target_index = 0;
          for (size_t dim = ndim; dim-- > 0;) {
            size_t const coordinate = remainder % intersection_count[dim];
            remainder /= intersection_count[dim];
            size_t const global_coordinate = intersection_start[dim]+coordinate;
            block_index += (global_coordinate-block_start[dim])*block_strides[dim];
            target_index += (global_coordinate-target_start[dim])*target_strides[dim];
          }
          std::memcpy(host.data()+target_index,decoded.data()+block_index*sizeof(T),sizeof(T));
        }
      }
      host.deep_copy_to(array);
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

    FileIO(MPI_Comm comm, std::string backend, int compression_min_bytes = 1048576) :
      backend(std::move(backend)), comm(comm), compression_min_bytes(compression_min_bytes) {
#ifdef PORTURB_HAS_ADIOS2
      if (this->backend != "adios2") {
        throw std::runtime_error("This portUrb build only supports the adios2 file_io_backend");
      }
      adios = std::make_unique<AdiosState>(comm,"porturb_"+std::to_string(adios_instance++));
#else
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
      bool const compression_candidate = !shape.empty() && !codec_dtype<T>().empty() &&
                                         total_bytes >= static_cast<size_t>(compression_min_bytes);
      auto const inserted = adios->variables.emplace(std::piecewise_construct,
                                                     std::forward_as_tuple(name),
                                                     std::forward_as_tuple(shape,std::type_index(typeid(T)),
                                                                           compression_candidate));
      if (!inserted.second) throw std::runtime_error("Duplicate ADIOS2 variable definition: "+name);
      if (!compression_candidate) define_ordinary_variable<T>(name);
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
      auto descriptor = adios->variables.find(variable_name);
      if (descriptor != adios->variables.end() && !descriptor->second.defined) {
        descriptor->second.attributes.emplace_back([this,value,variable_name,name] () {
          define_variable_attribute(value,variable_name,name);
        });
      } else {
        define_variable_attribute(value,variable_name,name);
      }
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
      auto &descriptor = adios->variables.at(name);
      auto variable = descriptor.defined ? adios_variable<T>(name) : define_ordinary_variable<T>(name);
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
      if (adios->io.InquireAttribute<std::string>("codec",name)) {
        adios_read_compressed(array,name,start);
        return;
      }
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
