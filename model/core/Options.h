
#pragma once

#include "main_header.h"
#include <variant>
#include <string>
#include <vector>
#include <unordered_map>

namespace core {

  class Options {
  public:
    using OptionValue = std::variant< short int,int,long int,long long int,
                                      unsigned short int,unsigned int,unsigned long int,unsigned long long int,
                                      float,double,long double,bool,char,std::string,
                                      std::vector<int>,std::vector<float>,std::vector<bool>,std::vector<double>,
                                      std::vector<size_t>,std::vector<std::string> >;
    Options()                          = default;
    ~Options()                         = default;
    Options(Options&&)                 = default;
    Options& operator=(Options&&)      = default;
    Options(const Options&)            = delete;
    Options& operator=(const Options&) = delete;

    template <class T> void add_option( std::string key , T value ) {
      if (key.empty()) return;
      options[std::move(key)] = OptionValue( std::in_place_type<T> , std::move(value) );
    }

    template <class T> void set_option( std::string key , T value ) { add_option(std::move(key), std::move(value)); }

    template <class T> T get_option(std::string const & key) const {
      auto it = options.find(key);
      if (it == options.end()) throw std::runtime_error(std::string("Option not found: ")+key);
      return std::get<T>(it->second);  // std::get throws std::bad_variant_access if the type is wrong
    }

    bool option_exists  (std::string const & key) const { return options.count(key) > 0; }
    void delete_option  (std::string const & key)       { options.erase(key); }
    int  get_num_options()                        const { return (int)options.size(); }
    void finalize       ()                              { options.clear(); }
    void clone_into     (Options & rhs)           const { rhs.options = options; }  // variant is copyable

  private:
    std::unordered_map<std::string,OptionValue> options;
  };

} // namespace core


