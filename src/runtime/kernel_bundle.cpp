#include "sycl/detail/kernel_bundle.hpp"

namespace sycl {

std::vector<kernel_id> get_kernel_ids() {
  std::vector<kernel_id> ids;
  const auto &kernels = detail::get_kernel_registry();
  for (const auto &[id, func] : kernels)
    ids.push_back(id);
  return ids;
}

bool is_compatible(const std::vector<kernel_id> &kernelIds, const device &dev) {
  for (const auto &kernel_id : kernelIds) {
    const auto &attributes = detail::get_kernel_attributes(kernel_id);
    for (const auto &attr : attributes) {
      using namespace detail;
      if (std::holds_alternative<kernel_attributes::reqd_work_group_size_1d>(
              attr)) {
        const auto &reqd =
            std::get<kernel_attributes::reqd_work_group_size_1d>(attr);
        if (reqd.x > dev.get_info<info::device::max_work_group_size>())
          return false;
      } else if (std::holds_alternative<
                     kernel_attributes::reqd_work_group_size_2d>(attr)) {
        const auto &reqd =
            std::get<kernel_attributes::reqd_work_group_size_2d>(attr);
        if (reqd.x * reqd.y > dev.get_info<info::device::max_work_group_size>())
          return false;
      } else if (std::holds_alternative<
                     kernel_attributes::reqd_work_group_size_3d>(attr)) {
        const auto &reqd =
            std::get<kernel_attributes::reqd_work_group_size_3d>(attr);
        if (reqd.x * reqd.y * reqd.z >
            dev.get_info<info::device::max_work_group_size>())
          return false;
      } else if (std::holds_alternative<kernel_attributes::reqd_sub_group_size>(
                     attr)) {
        const auto &reqd =
            std::get<kernel_attributes::reqd_sub_group_size>(attr);
        const auto &supported_sizes =
            dev.get_info<info::device::sub_group_sizes>();
        if (std::find(supported_sizes.begin(), supported_sizes.end(),
                      reqd.size) == supported_sizes.end())
          return false;
      } else if (std::holds_alternative<kernel_attributes::device_has>(attr)) {
        const auto &reqd = std::get<kernel_attributes::device_has>(attr);
        if (!dev.has(reqd.feature))
          return false;
      }
    }
  }
  return true;
}

kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input> &inputBundle,
        const std::vector<device> &devs, const property_list &propList) {
  return get_kernel_bundle<bundle_state::object>(inputBundle.get_context());
}

kernel_bundle<bundle_state::object>
compile(const kernel_bundle<bundle_state::input> &inputBundle,
        const property_list &propList) {
  return compile(inputBundle, inputBundle.get_devices(), propList);
}

kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &objectBundles,
     const std::vector<device> &devs, const property_list &propList) {
  if (devs.empty())
    throw sycl::exception{errc::invalid};
  return get_kernel_bundle<bundle_state::executable>(
      objectBundles[0].get_context());
}

kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &objectBundle,
     const property_list &propList) {
  return link({objectBundle}, objectBundle.get_devices(), propList);
}

kernel_bundle<bundle_state::executable>
link(const std::vector<kernel_bundle<bundle_state::object>> &objectBundles,
     const property_list &propList) {
  return link(objectBundles, objectBundles[0].get_devices(), propList);
}

kernel_bundle<bundle_state::executable>
link(const kernel_bundle<bundle_state::object> &objectBundle,
     const std::vector<device> &devs, const property_list &propList) {
  return link(std::vector{objectBundle}, devs, propList);
}

kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &inputBundle,
      const std::vector<device> &devs, const property_list &propList) {
  return link(compile(inputBundle, devs, propList), devs, propList);
}

kernel_bundle<bundle_state::executable>
build(const kernel_bundle<bundle_state::input> &inputBundle,
      const property_list &propList) {
  return build(inputBundle, inputBundle.get_devices(), propList);
}

} // namespace sycl
