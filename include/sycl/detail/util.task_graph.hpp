#pragma once

#include <future>
#include <ranges>
#include <vector>

#include "event.hpp"
#include "util.task.hpp"

namespace sycl::detail {

class task_graph {
public:
  event add_task(
      task t, std::vector<std::exception_ptr> &asyncExceptions,
      std::function<void()> throwAsyncExceptions = [] {}) {
    if (!t.m_action.has_value())
      return event{};

    std::unique_lock lock{m_mtx};

    for (int i = m_nodes.size() - 1; i >= 0; i--) {
      auto &node = m_nodes[i];
      auto ev_impl = detail::get_event_impl(node.m_event);
      if (ev_impl->get_status() == info::event_command_status::complete) {
        auto temp = std::move(m_nodes[i]);
        m_nodes.erase(m_nodes.begin() + i);
        lock.unlock();
        temp = {};
        lock.lock();
      }
    }

    for (const auto &req : t.m_requisites)
      for (const auto &node : std::views::reverse(m_nodes))
        if (req_depends_on_node(req, node)) {
          t.m_dependencies.push_back(node.m_event);
          break;
        }
    lock.unlock();

    auto ev = event{};
    auto ev_impl = detail::get_event_impl(ev);
    ev_impl->set_status(info::event_command_status::submitted);
    ev_impl->set_throw_async_exceptions(throwAsyncExceptions);
    ev_impl->set_dependencies([&] {
      std::vector<std::shared_ptr<detail::event_impl>> deps;
      for (const auto &dep_ev : t.m_dependencies)
        deps.push_back(detail::get_event_impl(dep_ev));
      return deps;
    }());

    auto result = std::async(
        std::launch::async,
        [this, ev, &asyncExceptions, dependencies = std::move(t.m_dependencies),
         action = std::move(t.m_action.value())]() {
          auto ev_impl = detail::get_event_impl(ev);
          event::wait(dependencies);
          ev_impl->set_status(info::event_command_status::running);
          try {
            action();
          } catch (...) {
            std::unique_lock lock{m_mtx};
            asyncExceptions.push_back(std::current_exception());
          }
          ev_impl->set_status(info::event_command_status::complete);
        });

    lock.lock();
    m_nodes.emplace_back(t, ev, std::move(result));
    return ev;
  }

private:
  struct node {
    task m_task;
    event m_event;
    std::future<void> m_future;
  };

  bool overlapping(const requisite &a, const requisite &b) {
    auto a_start = reinterpret_cast<std::uintptr_t>(a.m_target);
    auto a_end = a_start + a.m_size;
    auto b_start = reinterpret_cast<std::uintptr_t>(b.m_target);
    auto b_end = b_start + b.m_size;
    return ((a_start < b_end) && !(a_end < b_start)) ||
           ((b_start < a_end) && !(b_end < a_start));
  }

  bool req_depends_on_node(const requisite &req, const node &n) {
    auto effects = n.m_task.m_requisites;
    for (const auto &effect : effects)
      if (overlapping(req, effect))
        return true;
    return false;
  }

  std::mutex m_mtx;
  std::condition_variable m_cv;
  std::vector<node> m_nodes;
};

task_graph &get_task_graph();

} // namespace sycl::detail
