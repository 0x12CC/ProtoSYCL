#pragma once

// Prototype free-function-kernel form for the GCC/reflection path. There is no
// KHR spec for free-function kernels; this is a prototype with two pieces:
//
//   * SYCL_KHR_KERNEL(props...) - a pure P3394 annotation placed in front of a
//     free function. It carries khr compile-time properties and nothing else:
//     no kernel name, no tag type, no change to the function signature.
//
//   * SYCL_KHR_REGISTER_KERNELS(ns) - one per-translation-unit line (placed
//     after the kernel definitions) that reflects over namespace `ns`,
//     discovers every annotated free function, and registers it so it is
//     enumerable via sycl::get_kernel_ids(). Registration reads the properties
//     back via reflection and lowers them into the runtime attribute registry
//     that the handler enforcement path already consumes.
//
// Kernel identity is derived from the function itself (ffk_tag<^^F>), so no
// user-supplied name is needed anywhere.

#include <meta>
#include <optional>
#include <vector>

#include "../detail/kernel_bundle.hpp"       // sycl::get_kernel_id
#include "../detail/kernel_id.hpp"
#include "../detail/util.kernel_registry.hpp" // registry + set_kernel_attributes
#include "properties.hpp"

namespace sycl::khr::detail {

// A distinct empty type per reflected function, used as the KernelName tag so
// the existing type_info-keyed kernel_id machinery gives each free-function
// kernel a unique, stable identity without a user-supplied name.
template <std::meta::info F> struct ffk_tag {};

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

// True if the reflected member is a free function carrying at least one khr
// kernel property annotation.
consteval bool is_free_function_kernel(std::meta::info member) {
  namespace meta = std::meta;
  if (!meta::is_function(member))
    return false;
  for (auto annotation : meta::annotations_of(member)) {
    auto type = meta::remove_const(meta::type_of(meta::constant_of(annotation)));
    if (!meta::has_template_arguments(type))
      continue;
    auto tmpl = meta::template_of(type);
    if (tmpl == ^^sycl::khr::property::reqd_work_group_size_t ||
        tmpl == ^^sycl::khr::property::reqd_sub_group_size_t ||
        tmpl == ^^sycl::khr::property::work_group_size_hint_t)
      return true;
  }
  return false;
}

// Recursively collect all annotated free-function kernels reachable in the
// given namespace (including nested namespaces) within this translation unit.
consteval void collect_free_function_kernels(std::meta::info ns,
                                             std::vector<std::meta::info> &out) {
  namespace meta = std::meta;
  for (auto member : meta::members_of(ns, meta::access_context::current())) {
    if (meta::is_namespace(member)) {
      collect_free_function_kernels(member, out);
      continue;
    }
    if (is_free_function_kernel(member))
      out.push_back(member);
  }
}

template <std::meta::info Ns>
consteval std::vector<std::meta::info> free_function_kernels_of() {
  std::vector<std::meta::info> out;
  collect_free_function_kernels(Ns, out);
  return out;
}

// Register a single free-function kernel: make it enumerable via
// get_kernel_ids and populate its attribute list from reflection.
template <std::meta::info Fn>
sycl::kernel_id register_free_function_kernel() {
  const sycl::kernel_id id = sycl::get_kernel_id<ffk_tag<Fn>>();
  sycl::detail::get_kernel_registry().insert({id, typeid(ffk_tag<Fn>)});
  sycl::detail::set_kernel_attributes(typeid(ffk_tag<Fn>),
                                      read_free_function_attributes<Fn>());
  return id;
}

} // namespace sycl::khr::detail

namespace sycl::khr {

// Obtain the kernel_id of a free-function kernel from its reflection, e.g.
// sycl::khr::get_kernel_id<^^app::my_kernel>().
template <std::meta::info Fn> sycl::kernel_id get_kernel_id() {
  return sycl::get_kernel_id<detail::ffk_tag<Fn>>();
}

} // namespace sycl::khr

// Attach khr compile-time properties to the free function that follows. Pure
// annotation: no name, no signature wrapping. __VA_ARGS__ is forwarded verbatim
// into a single P3394 annotation, so a property whose template argument list
// contains commas (e.g. reqd_work_group_size<8, 4>) needs no extra parentheses.
//
//   SYCL_KHR_KERNEL(sycl::khr::property::reqd_work_group_size<8, 4>)
//   void my_kernel(sycl::nd_item<2> it) { /* ... */ }
#define SYCL_KHR_KERNEL(...) [[= __VA_ARGS__]]

// Register every annotated free-function kernel declared in namespace NS that
// is visible in this translation unit. Place once, after the kernel
// definitions (typically at the bottom of the TU). Runs at static init, so the
// kernels are enumerable before main.
#define SYCL_KHR_REGISTER_KERNELS(NS)                                          \
  namespace {                                                                  \
  [[maybe_unused]] const bool _sycl_khr_registered_##NS = [] {                 \
    template for (constexpr std::meta::info _sycl_khr_fn :                     \
                  std::define_static_array(                                    \
                      ::sycl::khr::detail::free_function_kernels_of<^^NS>()))  \
      ::sycl::khr::detail::register_free_function_kernel<_sycl_khr_fn>();      \
    return true;                                                               \
  }();                                                                         \
  }
