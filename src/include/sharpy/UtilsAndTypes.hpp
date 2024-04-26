// SPDX-License-Identifier: BSD-3-Clause

// convenience file/relict

#pragma once

#include <cstring>
#include <numeric>
#include <vector>

#include "CppTypes.hpp"
#include "PyTypes.hpp"

inline bool isText(std::string_view str) {
  return std::all_of(begin(str), end(str), [](auto c) {
    return std::isalnum(c) || std::ispunct(c) || std::isspace(c);
  });
};

inline bool get_bool_env(const std::string &name, bool default_value = 0) {
  auto envptr = getenv(name.c_str());
  if (envptr) {
    try {
      const std::string env(envptr);
      bool pos = env == "1" || env == "y" || env == "Y" || env == "on" ||
                 env == "ON" || env == "TRUE" || env == "True" || env == "true";
      bool neg = env == "0" || env == "n" || env == "N" || env == "off" ||
                 env == "OFF" || env == "FALSE" || env == "False" ||
                 env == "false";
      if (!pos && !neg) {
        throw std::invalid_argument("failed to parse boolean var");
      }
      return pos;
    } catch (...) {
      throw std::runtime_error("Invalid boolean environment variable: " + name);
    }
  }
  return default_value;
}

inline int get_int_env(const std::string &name, int default_value = 0) {
  auto envptr = getenv(name.c_str());
  if (envptr) {
    try {
      return std::stoi(envptr);
    } catch (...) {
      throw std::runtime_error("Invalid int environment variable: " + name);
    }
  }
  return default_value;
}

inline std::string get_text_env(const std::string &name,
                                const std::string &default_value = "") {
  auto envptr = getenv(name.c_str());
  if (envptr) {
    try {
      if (!isText(envptr)) {
        throw std::invalid_argument("invalid string");
      }
      return envptr;
    } catch (...) {
      throw std::runtime_error("Invalid text environment variable: " + name);
    }
  }
  return std::string(default_value);
}

inline bool useGPU() {
  auto device = get_text_env("SHARPY_DEVICE");
  return !(device.empty() || device == "host" || device == "cpu");
}

inline bool useCUDA() { return get_bool_env("SHARPY_USE_CUDA"); }
