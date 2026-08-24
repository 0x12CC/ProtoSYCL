#include <cstdarg>
#include <cxxabi.h>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "sycl/detail/util.kernel_registry.hpp"

namespace sycl::detail {

std::unordered_map<kernel_id, std::type_index> &get_kernel_registry() {
  static std::unordered_map<kernel_id, std::type_index> kernel_registry{};
  return kernel_registry;
}

std::string demangle_name(const std::string &mangled_name) {
  int status = 0;
  char *const demangled =
      abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status);
  if (status == 0 && demangled != nullptr) {
    std::string result(demangled);
    std::free(demangled);
    return result;
  }
  return mangled_name;
}

std::unordered_map<std::string, kernel_attribute_list> &
get_attribute_registry() {
  static std::unordered_map<std::string, kernel_attribute_list>
      attribute_registry{};
  return attribute_registry;
}

#if defined(PROTOSYCL_REFLECTION)
void set_kernel_attributes(const std::type_info &kernel_function,
                           kernel_attribute_list attributes) {
  const std::string key = demangle_name(kernel_function.name());
  get_attribute_registry()[key] = std::move(attributes);
}
#endif

kernel_attribute_list get_kernel_attributes(const kernel_id &id) {
  const auto &kernel_registry = get_kernel_registry();
  const auto &attribute_registry = get_attribute_registry();
  const auto it = kernel_registry.find(id);
  if (it != kernel_registry.end()) {
    const std::type_index &type_idx = it->second;
    const std::string kernel_name = type_idx.name();
    const std::string demangled_kernel_name = demangle_name(kernel_name);
    const auto attr_it = attribute_registry.find(demangled_kernel_name);
    if (attr_it != attribute_registry.end())
      return attr_it->second;
  }
  return {};
}

extern "C" void sycl_register_attribute(const char *name, const char *attribute,
                                        int arg_count, ...) {
  std::string demangled_name = demangle_name(name);
  // Remove trailing " const" if present.
  const std::string suffix = " const";
  if (demangled_name.ends_with(suffix))
    demangled_name =
        demangled_name.substr(0, demangled_name.size() - suffix.size());
  // Remove leading return type if present (e.g., "auto ", "void ").
  if (demangled_name.find("auto ") == 0)
    demangled_name = demangled_name.substr(5);
  else if (demangled_name.find("void ") == 0)
    demangled_name = demangled_name.substr(5);

  std::vector<std::uint64_t> args;
  va_list va_args;
  va_start(va_args, arg_count);
  for (int i = 0; i < arg_count; ++i) {
    std::uint64_t arg = va_arg(va_args, std::uint64_t);
    args.push_back(arg);
  }
  va_end(va_args);

  const std::string attribute_str{attribute};

  if (attribute_str == "sycl::reqd_work_group_size") {
    if (arg_count == 1) {
      kernel_attributes::reqd_work_group_size_1d attr{
          static_cast<std::size_t>(args[0])};
      get_attribute_registry()[demangled_name].push_back(attr);
      return;
    } else if (arg_count == 2) {
      kernel_attributes::reqd_work_group_size_2d attr{
          static_cast<std::size_t>(args[0]), static_cast<std::size_t>(args[1])};
      get_attribute_registry()[demangled_name].push_back(attr);
      return;
    } else if (arg_count == 3) {
      kernel_attributes::reqd_work_group_size_3d attr{
          static_cast<std::size_t>(args[0]), static_cast<std::size_t>(args[1]),
          static_cast<std::size_t>(args[2])};
      get_attribute_registry()[demangled_name].push_back(attr);
      return;
    }
  }

  if (attribute_str == "sycl::reqd_sub_group_size") {
    kernel_attributes::reqd_sub_group_size attr{
        static_cast<std::uint32_t>(args[0])};
    get_attribute_registry()[demangled_name].push_back(attr);
    return;
  }

  if (attribute_str == "sycl::device_has") {
    for (const auto aspect : args) {
      kernel_attributes::device_has attr{static_cast<sycl::aspect>(aspect)};
      get_attribute_registry()[demangled_name].push_back(attr);
    }
    return;
  }

  throw std::runtime_error("Unknown kernel attribute: " + attribute_str);
}

} // namespace sycl::detail
