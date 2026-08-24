#pragma once

#include <cstdint>
#include <typeinfo>
#include <unordered_map>
#include <variant>
#include <vector>

#if defined(PROTOSYCL_REFLECTION)
#include <meta>
#include <optional>
#endif

#include "aspect.hpp"
#include "kernel_id.hpp"

namespace sycl::detail {

namespace kernel_attributes {

struct reqd_work_group_size_1d {
  std::size_t x;
};

struct reqd_work_group_size_2d {
  std::size_t x;
  std::size_t y;
};

struct reqd_work_group_size_3d {
  std::size_t x;
  std::size_t y;
  std::size_t z;
};

struct reqd_sub_group_size {
  std::uint32_t size;
};

// work_group_size_hint has an implementation-defined effect per the SYCL spec,
// so it is stored but not enforced. rank selects how many of x/y/z are valid.
struct work_group_size_hint {
  std::size_t rank;
  std::size_t x;
  std::size_t y;
  std::size_t z;
};

struct device_has {
  sycl::aspect feature;
};

using attribute =
    std::variant<reqd_work_group_size_1d, reqd_work_group_size_2d,
                 reqd_work_group_size_3d, reqd_sub_group_size,
                 work_group_size_hint, device_has>;

} // namespace kernel_attributes

using kernel_attribute_list = std::vector<kernel_attributes::attribute>;

std::unordered_map<kernel_id, std::type_index> &get_kernel_registry();

kernel_attribute_list get_kernel_attributes(const kernel_id &id);

#if defined(PROTOSYCL_REFLECTION)
// Store a kernel's attribute list keyed by its function type (the runtime keys
// the attribute registry by the demangled type name; see get_kernel_attributes).
void set_kernel_attributes(const std::type_info &kernel_function,
                           kernel_attribute_list attributes);

// Read a single annotation of type Attr off a kernel functor via C++26
// reflection, if present. Annotations are looked for both on the functor type
// itself and on its (non-template) member functions, so `operator()` may carry
// the property directly. Note: g++ 16.1 cannot read annotations off a *template*
// member function, so a functor with a templated `operator()` must carry the
// annotation on the type instead.
template <typename KernelFunction, typename Attr>
consteval std::optional<Attr> read_kernel_annotation() {
  namespace meta = std::meta;
  for (auto a : meta::annotations_of(^^KernelFunction))
    if (meta::remove_const(meta::type_of(meta::constant_of(a))) == ^^Attr)
      return meta::extract<Attr>(meta::constant_of(a));
  for (auto member :
       meta::members_of(^^KernelFunction, meta::access_context::current()))
    if (meta::is_function(member))
      for (auto a : meta::annotations_of(member))
        if (meta::remove_const(meta::type_of(meta::constant_of(a))) == ^^Attr)
          return meta::extract<Attr>(meta::constant_of(a));
  return std::nullopt;
}

// Build the full attribute list for a kernel functor from its annotations.
template <typename KernelFunction>
kernel_attribute_list read_kernel_attributes() {
  using namespace kernel_attributes;
  kernel_attribute_list attributes;
  if (auto a = read_kernel_annotation<KernelFunction, reqd_work_group_size_1d>())
    attributes.push_back(*a);
  if (auto a = read_kernel_annotation<KernelFunction, reqd_work_group_size_2d>())
    attributes.push_back(*a);
  if (auto a = read_kernel_annotation<KernelFunction, reqd_work_group_size_3d>())
    attributes.push_back(*a);
  if (auto a = read_kernel_annotation<KernelFunction, reqd_sub_group_size>())
    attributes.push_back(*a);
  return attributes;
}
#endif // PROTOSYCL_REFLECTION

template <typename KernelName, typename KernelFunction>
kernel_id register_kernel() {
  const kernel_id id = get_kernel_id<KernelName>();
  get_kernel_registry().insert({id, typeid(KernelFunction)});
#if defined(PROTOSYCL_REFLECTION)
  // On the reflection path there is no compiler plugin to call
  // sycl_register_attribute; read the properties from the functor's annotations
  // and populate the attribute registry directly.
  set_kernel_attributes(typeid(KernelFunction),
                        read_kernel_attributes<KernelFunction>());
#endif
  return id;
}

extern "C" void sycl_register_attribute(const char *name, const char *attribute,
                                        int argCount, ...);

// The clang plugin recognizes this annotation to emit kernel-registration
// bookkeeping; on the GCC/reflection path the plugin is absent and the
// annotation would be an ignored (warned-about) attribute, so drop it there.
#if defined(PROTOSYCL_REFLECTION)
#define PROTOSYCL_KERNEL_REGISTRATION
#else
#define PROTOSYCL_KERNEL_REGISTRATION [[clang::annotate("sycl_kernel_registration")]]
#endif

template <typename KernelName, typename KernelFunction>
PROTOSYCL_KERNEL_REGISTRATION inline const sycl::kernel_id
    kernel_registry_v = register_kernel<KernelName, KernelFunction>();

} // namespace sycl::detail
