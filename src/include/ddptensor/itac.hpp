#pragma once

#ifndef NDEBUG

#define VT_ErrorCode_t int // it is actually an enum...
extern "C" {
VT_ErrorCode_t VT_classdef(const char *, int *) __attribute__((weak));
VT_ErrorCode_t VT_funcdef(const char *, int, int *) __attribute__((weak));
VT_ErrorCode_t VT_begin(int) __attribute__((weak));
VT_ErrorCode_t VT_end(int) __attribute__((weak));
}
#define VT(_sym, ...)                                                          \
  if (_sym != NULL) {                                                          \
    _sym(__VA_ARGS__);                                                         \
  }
#define HAS_ITAC() (VT_classdef != NULL)

#else

#define VT(...) ;
#define HAS_ITAC() false

#endif
