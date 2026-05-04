#pragma once

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

#include "backend.hpp"
#include "util.descriptor.hpp"

namespace sycl {

namespace info {

enum class event_command_status { submitted, running, complete };

namespace event {
struct command_execution_status : detail::descriptor<event_command_status> {};
} // namespace event

namespace event_profiling {
struct command_submit : detail::descriptor<std::uint64_t> {};
struct command_start : detail::descriptor<std::uint64_t> {};
struct command_end : detail::descriptor<std::uint64_t> {};
} // namespace event_profiling

} // namespace info

class event;

namespace detail {

class event_impl {
public:
  info::event_command_status get_status() const {
    std::scoped_lock lock{m_mtx};
    return m_status;
  }

  void set_status(info::event_command_status status) {
    std::unique_lock lock{m_mtx};
    m_status = status;
    lock.unlock();
    m_cv.notify_all();
  }

  void set_throw_async_exceptions(std::function<void()> func) {
    std::scoped_lock lock{m_mtx};
    m_throw_async_exceptions = std::move(func);
  }

  const std::vector<std::shared_ptr<event_impl>> &get_dependencies() const {
    std::scoped_lock lock{m_mtx};
    return m_dependencies;
  }

  void set_dependencies(const std::vector<std::shared_ptr<event_impl>> &deps) {
    std::scoped_lock lock{m_mtx};
    m_dependencies = deps;
  }

  void wait() {
    std::unique_lock lock{m_mtx};
    if (m_status != info::event_command_status::complete)
      m_cv.wait(lock, [&] {
        return m_status == info::event_command_status::complete;
      });
  }

  void wait_and_throw() {
    wait();
    m_throw_async_exceptions();
  }

private:
  mutable std::mutex m_mtx;
  std::condition_variable m_cv;
  info::event_command_status m_status{info::event_command_status::complete};
  std::vector<std::shared_ptr<event_impl>> m_dependencies;
  std::function<void()> m_throw_async_exceptions = [] {};
};

std::shared_ptr<detail::event_impl> get_event_impl(event);
event get_event(std::shared_ptr<detail::event_impl>);

} // namespace detail

class event {
public:
  event() {}

  /* -- common interface members -- */

  backend get_backend() const noexcept { return backend::proto_sycl; }

  std::vector<event> get_wait_list() {
    std::vector<event> dependencies;
    for (const auto &dep_impl : m_impl->get_dependencies())
      dependencies.push_back(detail::get_event(dep_impl));
    return dependencies;
  }

  void wait() { m_impl->wait(); }

  static void wait(const std::vector<event> &eventList) {
    std::for_each(eventList.begin(), eventList.end(),
                  [](event ev) { ev.wait(); });
  }

  void wait_and_throw() { m_impl->wait_and_throw(); }

  static void wait_and_throw(const std::vector<event> &eventList) {
    std::for_each(eventList.begin(), eventList.end(),
                  [](event ev) { ev.wait_and_throw(); });
  }

  template <typename Param> typename Param::return_type get_info() const;

  template <>
  typename info::event::command_execution_status::return_type
  get_info<info::event::command_execution_status>() const {
    return m_impl->get_status();
  }

  template <typename Param>
  typename Param::return_type get_backend_info() const;

  template <typename Param>
  typename Param::return_type get_profiling_info() const;

  template <>
  typename info::event_profiling::command_end::return_type
  get_profiling_info<info::event_profiling::command_submit>() const {
    return 0;
  }

  template <>
  typename info::event_profiling::command_end::return_type
  get_profiling_info<info::event_profiling::command_start>() const {
    return 0;
  }

  template <>
  typename info::event_profiling::command_end::return_type
  get_profiling_info<info::event_profiling::command_end>() const {
    return 0;
  }

  friend bool operator==(const event &lhs, const event &rhs) {
    return lhs.m_impl.get() == rhs.m_impl.get();
  }

private:
  friend std::hash<event>;
  friend std::shared_ptr<detail::event_impl> detail::get_event_impl(event);
  friend event detail::get_event(std::shared_ptr<detail::event_impl>);

  std::shared_ptr<detail::event_impl> m_impl{
      std::make_shared<detail::event_impl>()};
};

inline std::shared_ptr<detail::event_impl> detail::get_event_impl(event ev) {
  return ev.m_impl;
}

inline event detail::get_event(std::shared_ptr<detail::event_impl> impl) {
  event ev;
  ev.m_impl = impl;
  return ev;
}

} // namespace sycl

template <> struct std::hash<sycl::event> {
  std::size_t operator()(const sycl::event &event) const noexcept {
    return std::hash<decltype(event.m_impl)>{}(event.m_impl);
  }
};
