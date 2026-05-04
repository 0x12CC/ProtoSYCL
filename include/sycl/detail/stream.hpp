#pragma once

#include <iomanip>
#include <iostream>

#include "group.hpp"
#include "h_item.hpp"
#include "handler.hpp"
#include "id.hpp"
#include "nd_item.hpp"
#include "nd_range.hpp"
#include "property_list.hpp"
#include "range.hpp"
#include "vec.hpp"

namespace sycl {

enum class stream_manipulator {
  flush,
  dec,
  hex,
  oct,
  noshowbase,
  showbase,
  noshowpos,
  showpos,
  endl,
  fixed,
  scientific,
  hexfloat,
  defaultfloat
};

const stream_manipulator flush = stream_manipulator::flush;

const stream_manipulator dec = stream_manipulator::dec;

const stream_manipulator hex = stream_manipulator::hex;

const stream_manipulator oct = stream_manipulator::oct;

const stream_manipulator noshowbase = stream_manipulator::noshowbase;

const stream_manipulator showbase = stream_manipulator::showbase;

const stream_manipulator noshowpos = stream_manipulator::noshowpos;

const stream_manipulator showpos = stream_manipulator::showpos;

const stream_manipulator endl = stream_manipulator::endl;

const stream_manipulator fixed = stream_manipulator::fixed;

const stream_manipulator scientific = stream_manipulator::scientific;

const stream_manipulator hexfloat = stream_manipulator::hexfloat;

const stream_manipulator defaultfloat = stream_manipulator::defaultfloat;

inline auto setprecision(int precision) { return std::setprecision(precision); }

inline auto setw(int width) { return std::setw(width); }

class stream {
public:
  stream(std::size_t totalBufferSize, std::size_t workItemBufferSize,
         handler &cgh, const property_list &propList = {})
      : m_totalSize{totalBufferSize}, m_workItemSize{workItemBufferSize},
        m_handler{cgh}, m_props{propList} {}

  friend bool operator==(const stream &lhs, const stream &rhs) {
    return lhs.m_id == rhs.m_id;
  }

  /* -- property interface members -- */

  std::size_t size() const noexcept { return m_totalSize; }

  // Deprecated
  std::size_t get_size() const { return size(); }

  std::size_t get_work_item_buffer_size() const { return m_workItemSize; }

  /* get_max_statement_size() has the same functionality as
     get_work_item_buffer_size(), and is provided for backward compatibility.
     get_max_statement_size() is a deprecated query. */
  std::size_t get_max_statement_size() const {
    return get_work_item_buffer_size();
  }

private:
  template <typename T>
  friend const stream &operator<<(const stream &os, const T &rhs);
  friend std::hash<stream>;

  std::shared_ptr<std::monostate> m_id{std::make_shared<std::monostate>()};
  std::size_t m_totalSize;
  std::size_t m_workItemSize;
  std::reference_wrapper<handler> m_handler;
  property_list m_props;
};

template <typename T> const stream &operator<<(const stream &os, const T &rhs) {
  std::cout << rhs;
  return os;
}

const stream &operator<<(const stream &os, const stream_manipulator &rhs);

template <typename T, int N>
const stream &operator<<(const stream &os, const vec<T, N> &rhs) {
  os << "vec<" << typeid(T).name() << ", " << N << ">{";
  for (int i = 0; i < N - 1; i++)
    os << rhs[i] << ", ";
  os << rhs[N - 1];
  os << "}";
  return os;
}

template <int Dimensions>
const stream &operator<<(const stream &os, const id<Dimensions> &rhs) {
  os << "id<" << Dimensions << ">{";
  for (int i = 0; i < Dimensions - 1; i++)
    os << rhs[i] << ", ";
  os << rhs[Dimensions - 1];
  os << "}";
  return os;
}

template <int Dimensions>
const stream &operator<<(const stream &os, const range<Dimensions> &rhs) {
  os << "range<" << Dimensions << ">{";
  for (int i = 0; i < Dimensions - 1; i++)
    os << rhs[i] << ", ";
  os << rhs[Dimensions - 1];
  os << "}";
  return os;
}

template <int Dimensions>
const stream &operator<<(const stream &os, const nd_range<Dimensions> &rhs) {
  os << "nd_range<" << Dimensions << ">{";
  os << "global: " << rhs.get_global_range() << ", ";
  os << "local: " << rhs.get_local_range() << ", ";
  os << "group: " << rhs.get_group_range() << "}";
  return os;
}

template <int Dimensions, bool WithOffset>
const stream &operator<<(const stream &os,
                         const item<Dimensions, WithOffset> &rhs) {
  os << "item<" << Dimensions << ">{";
  for (int i = 0; i < Dimensions - 1; i++)
    os << rhs[i] << ", ";
  os << rhs[Dimensions - 1];
  os << "}";
  return os;
}

template <int Dimensions>
const stream &operator<<(const stream &os, const nd_item<Dimensions> &rhs) {
  os << "nd_item<" << Dimensions << ">{";
  os << "global_id: " << rhs.get_global_id() << ", ";
  os << "global_range: " << rhs.get_global_range() << ", ";
  os << "local_id: " << rhs.get_local_id() << ", ";
  os << "local_range: " << rhs.get_local_range() << "}";
  return os;
}

template <int Dimensions>
const stream &operator<<(const stream &os, const h_item<Dimensions> &rhs) {
  os << "h_item<" << Dimensions << ">{";
  os << "global: " << rhs.get_global() << ", ";
  os << "local: " << rhs.get_local() << ", ";
  os << "logical: " << rhs.get_logical_local() << "}";
  return os;
}

template <int Dimensions>
const stream &operator<<(const stream &os, const group<Dimensions> &rhs) {
  os << "group<" << Dimensions << ">{";
  os << "id: " << rhs.get_group_id() << ", ";
  os << "range: " << rhs.get_group_range() << "}";
  return os;
}

} // namespace sycl

template <> struct std::hash<sycl::stream> {
  std::size_t operator()(const sycl::stream &stream) const noexcept {
    return std::hash<decltype(stream.m_id)>{}(stream.m_id);
  }
};
