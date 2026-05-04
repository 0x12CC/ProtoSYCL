#pragma once

#include "event.hpp"

namespace sycl::detail {

struct requisite {
  enum class access_mode {
    read,
    write,
    read_write,
    no_init_write,
    no_init_read_write,
  };

  const void *m_target;
  const std::size_t m_size;
  access_mode m_mode;

  static access_mode merge_access_modes(access_mode a, access_mode b) {
    if (a == b)
      return a;
    if (a == access_mode::read && b == access_mode::write)
      return access_mode::read_write;
    if (a == access_mode::read && b == access_mode::read_write)
      return access_mode::read_write;
    if (a == access_mode::write && b == access_mode::read_write)
      return access_mode::read_write;
    if (a == access_mode::no_init_write && b == access_mode::no_init_read_write)
      return access_mode::no_init_read_write;
    if (a == access_mode::no_init_write && b == access_mode::write)
      return access_mode::write;
    if (a == access_mode::no_init_write && b == access_mode::read)
      return access_mode::read_write;
    if (a == access_mode::no_init_write && b == access_mode::read_write)
      return access_mode::read_write;
    if (a == access_mode::no_init_read_write && b == access_mode::write)
      return access_mode::read_write;
    if (a == access_mode::no_init_read_write && b == access_mode::read)
      return access_mode::read_write;
    if (a == access_mode::no_init_read_write && b == access_mode::read_write)
      return access_mode::read_write;
    return merge_access_modes(b, a);
  }
};

struct task {
  std::optional<std::function<void()>> m_action;
  std::vector<event> m_dependencies;
  std::vector<requisite> m_requisites;

  void add_requisite(const requisite &new_req) {
    for (requisite &req : m_requisites)
      if (new_req.m_target == req.m_target && new_req.m_size == req.m_size) {
        req.m_mode = requisite::merge_access_modes(new_req.m_mode, req.m_mode);
        return;
      }
    m_requisites.push_back(new_req);
  }

  std::vector<requisite> get_effects() const {
    std::vector<requisite> effects;
    for (const requisite &req : m_requisites)
      if (req.m_mode == requisite::access_mode::write ||
          req.m_mode == requisite::access_mode::read_write ||
          req.m_mode == requisite::access_mode::no_init_write ||
          req.m_mode == requisite::access_mode::no_init_read_write)
        effects.push_back(req);
    return effects;
  }
};

} // namespace sycl::detail
