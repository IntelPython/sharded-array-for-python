// SPDX-License-Identifier: BSD-3-Clause

// type promotion according to array API

#pragma once

namespace DDPT {

template <typename T, typename U = void, class Enable = void> struct promote {};

template <> struct promote<float> { using type = double; };
template <> struct promote<int64_t> { using type = double; };
template <> struct promote<int32_t> { using type = int64_t; };
template <> struct promote<int16_t> { using type = int32_t; };
template <> struct promote<int8_t> { using type = int16_t; };
template <> struct promote<uint64_t> { using type = double; };
template <> struct promote<uint32_t> { using type = uint64_t; };
template <> struct promote<uint16_t> { using type = uint32_t; };
template <> struct promote<uint8_t> { using type = uint16_t; };

// both are T -> T
template <typename T, typename U>
struct promote<T, U, typename std::enable_if<std::is_same<T, U>::value>::type> {
  using type = T;
};

// one is double, the other something else -> double
template <typename T>
struct promote<T, double,
               typename std::enable_if<!std::is_same<T, double>::value>::type> {
  using type = double;
};

// reverse: one is double, the other something else -> double
template <typename T>
struct promote<double, T,
               typename std::enable_if<!std::is_same<T, double>::value>::type> {
  using type = double;
};

// one is float, the other integral -> float
template <typename T>
struct promote<T, float,
               typename std::enable_if<std::is_integral<T>::value>::type> {
  using type = float;
};

// reverse: one is float, the other integral -> float
template <typename T>
struct promote<float, T,
               typename std::enable_if<std::is_integral<T>::value>::type> {
  using type = float;
};

// both are integral, T is larger, T is signed or both unsigned -> T
template <typename T, typename U>
struct promote<
    T, U,
    typename std::enable_if<
        std::is_integral<T>::value && std::is_integral<U>::value &&
        (sizeof(T) > sizeof(U)) &&
        (std::is_signed<T>::value || std::is_unsigned<U>::value)>::type> {
  using type = T;
};

// both are integral, T is larger, U is signed and T is unsigned ->
// promote(signed(T))
template <typename T, typename U>
struct promote<
    T, U,
    typename std::enable_if<
        std::is_integral<T>::value && std::is_integral<U>::value &&
        (sizeof(T) > sizeof(U)) &&
        (std::is_signed<U>::value && std::is_unsigned<T>::value)>::type> {
  using type = typename promote<typename std::make_signed<T>::type>::type;
};

// both are integral, U is larger, U is signed or both unsigned -> U
template <typename T, typename U>
struct promote<
    T, U,
    typename std::enable_if<
        std::is_integral<T>::value && std::is_integral<U>::value &&
        (sizeof(T) < sizeof(U)) &&
        (std::is_signed<U>::value || std::is_unsigned<T>::value)>::type> {
  using type = U;
};

// both are integral, U is larger, T is signed and U is unsigned ->
// promote(signed(U))
template <typename T, typename U>
struct promote<
    T, U,
    typename std::enable_if<
        std::is_integral<T>::value && std::is_integral<U>::value &&
        (sizeof(T) < sizeof(U)) &&
        (std::is_signed<T>::value && std::is_unsigned<U>::value)>::type> {
  using type = typename promote<typename std::make_signed<U>::type>::type;
};

// both are integral and have same size, T is signed, U is unsigned ->
// promote(signed(T))
template <typename T, typename U>
struct promote<
    T, U,
    typename std::enable_if<
        std::is_integral<T>::value && std::is_integral<U>::value &&
        (sizeof(T) == sizeof(U)) &&
        (std::is_signed<T>::value && std::is_unsigned<U>::value)>::type> {
  using type = typename promote<T>::type;
};

// both are integral and have same size, U is signed, T is unsigned ->
// promote(signed(U))
template <typename T, typename U>
struct promote<
    T, U,
    typename std::enable_if<
        std::is_integral<T>::value && std::is_integral<U>::value &&
        (sizeof(T) == sizeof(U)) &&
        (std::is_signed<U>::value && std::is_unsigned<T>::value)>::type> {
  using type = typename promote<U>::type;
};
} // namespace DDPT
