#include "sycl/detail/util.task_graph.hpp"

namespace sycl::detail {

task_graph &get_task_graph() {
  static task_graph graph{};
  return graph;
}

} // namespace sycl::detail
