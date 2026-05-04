#pragma once

namespace sycl {

enum class target {
  device,
  host_task,
  constant_buffer,        // Deprecated
  local,                  // Deprecated
  host_buffer,            // Deprecated
  image,                  // Not in spec (removed)
  host_image,             // Not in spec (removed)
  image_array,            // Not in spec (removed)
  global_buffer = device, // Deprecated
};

enum class access_mode {
  read,
  write,
  read_write,
  discard_write,      // Deprecated in SYCL 2020
  discard_read_write, // Deprecated in SYCL 2020
  atomic              // Deprecated in SYCL 2020
};

namespace access {

enum class address_space {
  global_space,
  local_space,
  constant_space, // Deprecated in SYCL 2020
  private_space,
  generic_space
};

enum class decorated {
  no,
  yes,
  legacy // Deprecated in SYCL 2020
};

// The legacy type "access::target" is deprecated.
using target = sycl::target;

enum class placeholder { // Deprecated
  false_t,
  true_t
};

// The legacy type "access::mode" is deprecated.
using mode = sycl::access_mode;

} // namespace access

} // namespace sycl
