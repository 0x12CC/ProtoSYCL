#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "backend.hpp"
#include "context.hpp"
#include "device_image.hpp"
#include "exception.hpp"
#include "kernel.hpp"
#include "kernel_id.hpp"
#include "specialization_id.hpp"
#include "util.kernel_registry.hpp"

namespace sycl::detail {

template <bundle_state State> struct kernel_bundle_impl {
  context m_ctx;
  std::vector<device> m_devices;
  std::vector<kernel_id> m_kernels;
  std::vector<device_image<State>> m_images;
  std::unordered_map<const void *, std::any> m_specializationConstants;
};

} // namespace sycl::detail

namespace sycl {

template <bundle_state State> class kernel_bundle {
public:
  using device_image_iterator =
      std::vector<device_image<State>>::const_iterator;

  friend bool operator==(const kernel_bundle &,
                         const kernel_bundle &) = default;

  bool empty() const noexcept { return m_impl->m_images.empty(); }

  backend get_backend() const noexcept { return m_impl->m_ctx.get_backend(); }

  context get_context() const noexcept { return m_impl->m_ctx; }
  std::vector<device> get_devices() const noexcept { return m_impl->m_devices; }

  bool has_kernel(const kernel_id &kernelId) const noexcept {
    return std::find(m_impl->m_kernels.begin(), m_impl->m_kernels.end(),
                     kernelId) != m_impl->m_kernels.end();
  }

  bool has_kernel(const kernel_id &kernelId, const device &dev) const noexcept {
    return has_kernel(kernelId);
  }

  template <typename KernelName> bool has_kernel() const noexcept {
    return has_kernel(get_kernel_id<KernelName>());
  }

  template <typename KernelName>
  bool has_kernel(const device &dev) const noexcept {
    return has_kernel(get_kernel_id<KernelName>(), dev);
  }

  std::vector<kernel_id> get_kernel_ids() const { return m_impl->m_kernels; }

  kernel get_kernel(const kernel_id &kernelId) const
    requires(State == bundle_state::executable)
  {
    return kernel{kernelId, m_impl->m_ctx};
  }

  template <typename KernelName>
  kernel get_kernel() const
    requires(State == bundle_state::executable)
  {
    return get_kernel(get_kernel_id<KernelName>());
  }

  bool contains_specialization_constants() const noexcept {
    return m_impl->m_specializationConstants.size() > 0;
  }

  bool native_specialization_constant() const noexcept { return false; }

  template <auto &SpecName> bool has_specialization_constant() const noexcept {
    return m_impl->m_specializationConstants.contains(
        static_cast<const void *>(&SpecName));
  }

  template <auto &SpecName>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type value)
    requires(State == bundle_state::input)
  {
    const void *index = static_cast<const void *>(&SpecName);
    m_impl->m_specializationConstants[index] = value;
  }

  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() const {
    const void *index = static_cast<const void *>(&SpecName);
    if (m_impl->m_specializationConstants.contains(index))
      return std::any_cast<
          typename std::remove_reference_t<decltype(SpecName)>::value_type>(
          m_impl->m_specializationConstants.at(index));
    return detail::get_specialization_default_value(SpecName);
  }

  device_image_iterator begin() const { return m_impl->m_images.begin(); }

  device_image_iterator end() const { return m_impl->m_images.end(); }

private:
  friend std::hash<kernel_bundle>;
  friend class handler;

public:
  kernel_bundle(context ctx, std::vector<device> devices,
                std::vector<kernel_id> kernels,
                std::vector<device_image<State>> images) {
    m_impl->m_ctx = ctx;
    m_impl->m_devices = std::move(devices);
    m_impl->m_kernels = std::move(kernels);
    m_impl->m_images = std::move(images);
  }

private:
  std::shared_ptr<detail::kernel_bundle_impl<State>> m_impl{
      std::make_shared<detail::kernel_bundle_impl<State>>()};
};

bool is_compatible(const std::vector<kernel_id> &kernelIds, const device &dev);

template <typename KernelName> kernel_id get_kernel_id() {
  return kernel_id{typeid(std::type_identity<KernelName>)};
}

std::vector<kernel_id> get_kernel_ids();

template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt,
                                       const std::vector<device> &devs) {
  if (devs.empty())
    throw sycl::exception{errc::invalid, "device list empty"};
  for (const auto &dev : devs) {
    if constexpr (State == bundle_state::input)
      if (!dev.has(aspect::online_compiler))
        throw sycl::exception{errc::invalid,
                              "device doesn't support online compiler"};
    if constexpr (State == bundle_state::object)
      if (!dev.has(aspect::online_linker))
        throw sycl::exception{errc::invalid,
                              "device doesn't support online compiler"};
    if (std::find(ctxt.get_devices().begin(), ctxt.get_devices().end(), dev) ==
        ctxt.get_devices().end())
      throw sycl::exception{errc::invalid, "context doesn't include device"};
  }
  auto ids = get_kernel_ids();
  for (int i = ids.size() - 1; i >= 0; --i)
    if (!is_compatible({ids[i]}, device{}))
      ids.erase(ids.begin() + i);
  if (ids.empty())
    return kernel_bundle<State>{ctxt, devs, ids, {}};
  return kernel_bundle<State>{
      ctxt, devs, ids, {detail::create_device_image<State>()}};
}

template <bundle_state State>
kernel_bundle<State>
get_kernel_bundle(const context &ctxt, const std::vector<device> &devs,
                  const std::vector<kernel_id> &kernelIds) {
  if (devs.empty())
    throw sycl::exception{errc::invalid, "device list empty"};
  for (const auto &dev : devs) {
    if constexpr (State == bundle_state::input)
      if (!dev.has(aspect::online_compiler))
        throw sycl::exception{errc::invalid,
                              "device doesn't support online compiler"};
    if constexpr (State == bundle_state::object)
      if (!dev.has(aspect::online_linker))
        throw sycl::exception{errc::invalid,
                              "device doesn't support online compiler"};
    if (std::find(ctxt.get_devices().begin(), ctxt.get_devices().end(), dev) ==
        ctxt.get_devices().end())
      throw sycl::exception{errc::invalid, "context doesn't include device"};
  }
  for (const auto &id : kernelIds)
    if (!is_compatible({id}, device{}))
      throw sycl::exception{errc::invalid,
                            "kernel is not compatible with device"};
  if (kernelIds.empty())
    return kernel_bundle<State>{ctxt, devs, kernelIds, {}};
  return kernel_bundle<State>{
      ctxt, devs, kernelIds, {detail::create_device_image<State>()}};
}

template <bundle_state State, typename Selector>
kernel_bundle<State> get_kernel_bundle(const context &ctxt,
                                       const std::vector<device> &devs,
                                       Selector selector) {
  if (devs.empty())
    throw sycl::exception{errc::invalid};
  for (const auto &dev : devs) {
    if constexpr (State == bundle_state::input)
      if (!dev.has(aspect::online_compiler))
        throw sycl::exception{errc::invalid};
    if constexpr (State == bundle_state::object)
      if (!dev.has(aspect::online_linker))
        throw sycl::exception{errc::invalid};
    if (std::find(ctxt.get_devices().begin(), ctxt.get_devices().end(), dev) ==
        ctxt.get_devices().end())
      throw sycl::exception{errc::invalid};
  }
  auto image = detail::create_device_image<State>();
  if (selector(image))
    return kernel_bundle<State>{ctxt, devs, get_kernel_ids(), {image}};
  return kernel_bundle<State>{ctxt, devs, {}, {}};
}

template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt) {
  return get_kernel_bundle<State>(ctxt, ctxt.get_devices());
}

template <bundle_state State>
kernel_bundle<State>
get_kernel_bundle(const context &ctxt,
                  const std::vector<kernel_id> &kernelIds) {
  return get_kernel_bundle<State>(ctxt, ctxt.get_devices(), kernelIds);
}

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt) {
  return get_kernel_bundle<State>(ctxt, ctxt.get_devices(),
                                  {get_kernel_id<KernelName>()});
}

template <typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt,
                                       const std::vector<device> &devs) {
  return get_kernel_bundle<State>(ctxt, devs, {get_kernel_id<KernelName>()});
}

template <bundle_state State, typename Selector>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, Selector selector) {
  return get_kernel_bundle<State>(ctxt, ctxt.get_devices(), selector);
}

template <bundle_state State> bool has_kernel_bundle(const context &ctxt) {
  return has_kernel_bundle<State>(ctxt, ctxt.get_devices());
}

template <bundle_state State>
bool has_kernel_bundle(const context &ctxt,
                       const std::vector<kernel_id> &kernelIds) {
  return has_kernel_bundle<State>(ctxt, ctxt.get_devices(), kernelIds);
}

template <typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &ctxt) {
  return has_kernel_bundle<State>(ctxt, {get_kernel_id<KernelName>()});
}

template <bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs) {
  if (devs.empty())
    throw sycl::exception{errc::invalid, "device list empty"};
  if (get_kernel_ids().empty())
    return false;
  for (const auto &device : devs) {
    if constexpr (State == bundle_state::input)
      if (!device.has(aspect::online_compiler))
        return false;
    if constexpr (State == bundle_state::object)
      if (!device.has(aspect::online_linker))
        return false;

    const auto &kernels = detail::get_kernel_registry();
    for (const auto &[id, type_index] : kernels) {
      if (!sycl::is_compatible({id}, device))
        return false;
    }
  }
  return true;
}

template <bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs,
                       const std::vector<kernel_id> &kernelIds) {
  if (devs.empty())
    throw sycl::exception{errc::invalid, "device list empty"};
  if (kernelIds.empty())
    return true;
  for (const auto &device : devs) {
    if constexpr (State == bundle_state::input)
      if (!device.has(aspect::online_compiler))
        return false;
    if constexpr (State == bundle_state::object)
      if (!device.has(aspect::online_linker))
        return false;
    if (!is_compatible(kernelIds, device))
      return false;
  }
  return true;
}

template <typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs) {
  return has_kernel_bundle<State>(ctxt, devs, {get_kernel_id<KernelName>()});
}

bool is_compatible(const std::vector<kernel_id> &kernelIds, const device &dev);

template <typename KernelName> bool is_compatible(const device &dev) {
  return is_compatible({get_kernel_id<KernelName>()}, dev);
}

template <bundle_state State>
kernel_bundle<State> join(const std::vector<kernel_bundle<State>> &bundles) {
  std::vector<device> devices;
  std::vector<device_image<State>> images;
  std::vector<kernel_id> kernels;
  for (const auto &bundle : bundles) {
    for (const auto &dev : bundle.get_devices())
      if (std::find(devices.begin(), devices.end(), dev) == devices.end())
        devices.push_back(dev);
    for (const auto &img : bundle)
      if (std::find(images.begin(), images.end(), img) == images.end())
        images.push_back(img);
    for (const auto &kid : bundle.get_kernel_ids())
      if (std::find(kernels.begin(), kernels.end(), kid) == kernels.end())
        kernels.push_back(kid);
  }
  return kernel_bundle<State>{context{}, devices, kernels, images};
}

kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input> &inputBundle,
        const std::vector<device> &devs, const property_list &propList = {});

kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input> &inputBundle,
        const property_list &propList = {});

kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &objectBundles,
     const std::vector<device> &devs, const property_list &propList = {});

kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &objectBundle,
     const property_list &propList = {});

kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &objectBundles,
     const property_list &propList = {});

kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &objectBundle,
     const std::vector<device> &devs, const property_list &propList = {});

kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &inputBundle,
      const std::vector<device> &devs, const property_list &propList = {});

kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &inputBundle,
      const property_list &propList = {});

} // namespace sycl

template <sycl::bundle_state State>
struct std::hash<sycl::kernel_bundle<State>> {
  std::size_t
  operator()(const sycl::kernel_bundle<State> &kernel_bundle) const noexcept {
    return std::hash<decltype(kernel_bundle.m_impl)>{}(kernel_bundle.m_impl);
  }
};
