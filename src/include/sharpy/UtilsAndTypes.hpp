// SPDX-License-Identifier: BSD-3-Clause

// convenience file/relict

#pragma once

#include <cstring>
#include <numeric>
#include <vector>

#include "CppTypes.hpp"
#include "PyTypes.hpp"

inline bool isText(const std::string &str) {
  return std::all_of(begin(str), end(str), [](auto c) {
    return std::isalnum(c) || std::ispunct(c) || std::isspace(c);
  });
};

inline bool get_bool_env(const char *name, bool default_value = 0) {
  const char *envptr = getenv(name);
  if (envptr != nullptr) {
    try {
      auto c = std::string(envptr);
      bool pos = c == "1" || c == "y" || c == "Y" || c == "on" || c == "ON" ||
                 c == "TRUE" || c == "True" || c == "true";
      bool neg = c == "0" || c == "n" || c == "N" || c == "off" || c == "OFF" ||
                 c == "FALSE" || c == "False" || c == "false";
      if (!pos && !neg) {
        throw std::invalid_argument("failed to parse boolean var");
      }
      return pos;
    } catch (...) {
      std::string msg("Invalid boolean environment variable: ");
      msg += name;
      throw std::runtime_error(msg);
    }
  }
  return default_value;
}

inline int get_int_env(const char *name, int default_value = 0) {
  const char *envptr = getenv(name);
  if (envptr != nullptr) {
    try {
      return std::stoi(envptr);
    } catch (...) {
      std::string msg("Invalid int environment variable: ");
      msg += name;
      throw std::runtime_error(msg);
    }
  }
  return default_value;
}

inline std::string get_text_env(const char *name,
                                const char *default_value = "") {
  const char *envptr = getenv(name);
  if (envptr != nullptr) {
    try {
      std::string out(envptr);
      if (!out.empty() && !isText(out)) {
        throw std::invalid_argument("invalid string");
      }
      return out;
    } catch (...) {
      std::string msg("Invalid text environment variable: ");
      msg += name;
      throw std::runtime_error(msg);
    }
  }
  return std::string(default_value);
}

inline bool useGPU() {
  std::string device = get_text_env("SHARPY_DEVICE");
  return !(device.empty() || device == "host" || device == "cpu");
}
