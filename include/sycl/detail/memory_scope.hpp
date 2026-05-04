#pragma once

namespace sycl {

enum class memory_scope {
  work_item,
  sub_group,
  work_group,
  device,
  system
};

}
