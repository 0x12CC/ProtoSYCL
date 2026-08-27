#pragma once

#include "../handler.hpp"
#include "../queue.hpp"

#define SYCL_KHR_KERNEL()

namespace sycl::khr {

template <auto *Func> struct kernel_function_s {
  static constexpr auto value = Func;
};

template <auto *Func> inline constexpr kernel_function_s<Func> kernel_function;

template <auto *Func, typename... Args>
void launch_task(const queue &q, kernel_function_s<Func> k, Args &&...args) {
  queue{q}.single_task([=] { k.value(args...); }, std::forward<Args>(args)...);
}

template <auto *Func, typename... Args>
void launch_task(handler &h, kernel_function_s<Func> k, Args &&...args) {
  h.single_task([=] { k.value(args...); }, std::forward<Args>(args)...);
}

template <auto *Func, int Dims, typename... Args>
void launch_grouped(const queue &q, range<Dims> global, range<Dims> local,
                    kernel_function_s<Func> k, Args &&...args) {
  queue{q}.parallel_for(nd_range<Dims>{global, local},
                        [=](nd_item<Dims> item) { k.value(args...); });
}

template <auto *Func, int Dims, typename... Args>
void launch_grouped(handler &h, range<Dims> global, range<Dims> local,
                    kernel_function_s<Func> k, Args &&...args) {
  h.parallel_for(
      nd_range<Dims>{global, local},
      [=](nd_item<Dims> item) { k.value(args...); },
      std::forward<Args>(args)...);
}

} // namespace sycl::khr
