#pragma once

#include <cstdint>

#include "id.hpp"
#include "memory_scope.hpp"
#include "range.hpp"

namespace sycl {

class sub_group;

namespace detail {
sub_group make_sub_group(std::uint32_t, std::uint32_t, std::uint32_t,
                         std::uint32_t, std::uint32_t);
} // namespace detail

class sub_group {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = std::uint32_t;
  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  friend bool operator==(const sub_group &lhs, const sub_group &rhs) = default;

  id<1> get_group_id() const { return m_group_id; }

  id<1> get_local_id() const { return m_local_id; }

  range<1> get_local_range() const { return m_local_range; }

  range<1> get_group_range() const { return m_group_range; }

  range<1> get_max_local_range() const { return m_max_local_range; }

  std::uint32_t get_group_linear_id() const { return get_group_id()[0]; }

  std::uint32_t get_local_linear_id() const { return get_local_id()[0]; }

  std::uint32_t get_group_linear_range() const { return get_group_range()[0]; }

  std::uint32_t get_local_linear_range() const { return get_local_range()[0]; }

  bool leader() const { return get_local_linear_id() == 0; }

private:
  sub_group(std::uint32_t local_id, std::uint32_t local_range,
            std::uint32_t group_id, std::uint32_t group_range,
            std::uint32_t max_local_range)
      : m_local_id{local_id}, m_local_range{local_range}, m_group_id{group_id},
        m_group_range{group_range}, m_max_local_range{max_local_range} {}

  friend sub_group sycl::detail::make_sub_group(std::uint32_t, std::uint32_t,
                                                std::uint32_t, std::uint32_t,
                                                std::uint32_t);

  std::uint32_t m_local_id;
  std::uint32_t m_local_range;
  std::uint32_t m_group_id;
  std::uint32_t m_group_range;
  std::uint32_t m_max_local_range;
};

inline sub_group detail::make_sub_group(std::uint32_t local_id,
                                        std::uint32_t local_range,
                                        std::uint32_t group_id,
                                        std::uint32_t group_range,
                                        std::uint32_t max_local_range) {
  return sub_group{local_id, local_range, group_id, group_range,
                   max_local_range};
}

} // namespace sycl
