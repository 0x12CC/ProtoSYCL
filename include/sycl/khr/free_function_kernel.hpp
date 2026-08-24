#pragma once

// Prototype free-function-kernel form for the GCC/reflection path. There is no
// KHR spec for free-function kernels; the SYCL_KHR_KERNEL macro below is a
// prototype that (a) attaches a khr compile-time property to a free function as
// a P3394 annotation and (b) emits a static registrar so the kernel is
// enumerable via sycl::get_kernel_ids(). The property is read back from the
// function's annotation via C++26 reflection, reusing the runtime attribute
// registry that the handler enforcement path already consumes.

#include <meta>
#include <optional>

#include "../detail/kernel_bundle.hpp"       // sycl::get_kernel_id
#include "../detail/kernel_id.hpp"
#include "../detail/util.kernel_registry.hpp" // registry + set_kernel_attributes
#include "properties.hpp"

namespace sycl::khr::detail {

// A rank + up-to-3 dimensions read from a compile-time property whose values
// live in its template arguments.
struct dims_value {
  std::size_t rank;
  std::size_t dims[3];
};

// Read the (single) annotation on the free function Fn whose type is a
// specialization of the property template PropTemplate, returning its template
// arguments as dimensions. The values live in the type's template arguments, so
// no splicing of the (non-constexpr) loop variable is needed.
template <std::meta::info Fn, std::meta::info PropTemplate>
consteval std::optional<dims_value> read_dims_property() {
  namespace meta = std::meta;
  for (auto annotation : meta::annotations_of(Fn)) {
    auto type = meta::remove_const(meta::type_of(meta::constant_of(annotation)));
    if (meta::has_template_arguments(type) &&
        meta::template_of(type) == PropTemplate) {
      auto args = meta::template_arguments_of(type);
      dims_value value{args.size(), {0, 0, 0}};
      for (std::size_t i = 0; i < args.size(); ++i)
        value.dims[i] = meta::extract<std::size_t>(args[i]);
      return value;
    }
  }
  return std::nullopt;
}

// Build the runtime attribute list for a free-function kernel from its khr
// compile-time property annotations. Covers the core kernel-placed attributes;
// device_has (transitive) is intentionally not handled here.
template <std::meta::info Fn>
sycl::detail::kernel_attribute_list read_free_function_attributes() {
  using namespace sycl::detail::kernel_attributes;
  sycl::detail::kernel_attribute_list attributes;

  constexpr auto wgs =
      read_dims_property<Fn, ^^sycl::khr::property::reqd_work_group_size_t>();
  if constexpr (wgs.has_value()) {
    if constexpr (wgs->rank == 1)
      attributes.push_back(reqd_work_group_size_1d{wgs->dims[0]});
    else if constexpr (wgs->rank == 2)
      attributes.push_back(reqd_work_group_size_2d{wgs->dims[0], wgs->dims[1]});
    else if constexpr (wgs->rank == 3)
      attributes.push_back(
          reqd_work_group_size_3d{wgs->dims[0], wgs->dims[1], wgs->dims[2]});
  }

  constexpr auto sgs =
      read_dims_property<Fn, ^^sycl::khr::property::reqd_sub_group_size_t>();
  if constexpr (sgs.has_value())
    attributes.push_back(
        reqd_sub_group_size{static_cast<std::uint32_t>(sgs->dims[0])});

  constexpr auto hint =
      read_dims_property<Fn, ^^sycl::khr::property::work_group_size_hint_t>();
  if constexpr (hint.has_value())
    attributes.push_back(work_group_size_hint{hint->rank, hint->dims[0],
                                              hint->dims[1], hint->dims[2]});

  return attributes;
}

// Register a free-function kernel: make it enumerable via get_kernel_ids and
// populate its attribute list from reflection. KernelName is a per-kernel tag
// type; Fn is the reflection of the kernel function.
template <typename KernelName, std::meta::info Fn>
sycl::kernel_id register_free_function_kernel() {
  const sycl::kernel_id id = sycl::get_kernel_id<KernelName>();
  sycl::detail::get_kernel_registry().insert({id, typeid(KernelName)});
  sycl::detail::set_kernel_attributes(typeid(KernelName),
                                      read_free_function_attributes<Fn>());
  return id;
}

} // namespace sycl::khr::detail

// Strips the protective parentheses around the property argument (needed so a
// property whose template argument list contains commas, e.g.
// reqd_work_group_size<8, 4>, survives macro argument splitting).
#define SYCL_KHR_DETAIL_STRIP(...) __VA_ARGS__

// Declare a free-function kernel NAME with the (parenthesized) khr compile-time
// property PROP and the given parameter list. Usage:
//
//   SYCL_KHR_KERNEL(my_kernel,
//                   (sycl::khr::property::reqd_work_group_size<4>),
//                   sycl::nd_item<1> it) {
//     /* body */
//   }
//
// Expands to an annotated forward declaration, a static self-registrar, and the
// header of the definition that follows (the user supplies the body).
#define SYCL_KHR_KERNEL(NAME, PROP, ...)                                       \
  struct NAME##_KernelName {};                                                 \
  [[= SYCL_KHR_DETAIL_STRIP PROP]] void NAME(__VA_ARGS__);                     \
  [[maybe_unused]] static const ::sycl::kernel_id NAME##_khr_kid =             \
      ::sycl::khr::detail::register_free_function_kernel<NAME##_KernelName,     \
                                                         ^^NAME>();            \
  void NAME(__VA_ARGS__)
