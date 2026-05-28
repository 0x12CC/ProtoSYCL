#pragma once

#include <queue>
#include <unordered_map>

#include "access.hpp"
#include "accessor.hpp"
#include "event.hpp"
#include "exception.hpp"
#include "id.hpp"
#include "item.hpp"
#include "kernel.hpp"
#include "kernel_bundle.hpp"
#include "nd_item.hpp"
#include "nd_range.hpp"
#include "range.hpp"
#include "specialization_id.hpp"
#include "util.kernel_registry.hpp"
#include "util.task.hpp"

namespace sycl {

class handler;

using constant_map = std::unordered_map<const void *, std::any>;
using constant_map_ptr = std::shared_ptr<constant_map>;

class kernel_handler {
public:
  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() {
    const void *index = static_cast<const void *>(&SpecName);
    return m_constants->contains(index)
               ? std::any_cast<typename std::remove_reference_t<
                     decltype(SpecName)>::value_type>(m_constants->at(index))
               : detail::get_specialization_default_value(SpecName);
  }

private:
  kernel_handler(const constant_map_ptr constants) : m_constants{constants} {}

  friend handler;
  constant_map_ptr m_constants;
};

namespace detail {

struct unnamed_kernel;

template <typename DataT, int Dimensions>
void registerAccessor(handler &, const local_accessor<DataT, Dimensions> &);

template <int Dimensions, typename LaunchFn>
void nd_launch(nd_range<Dimensions> executionRange, const LaunchFn &launch)
  requires(1 <= Dimensions && Dimensions <= 3)
{
  const auto global_offset = executionRange.get_offset();
  const auto global_range = executionRange.get_global_range();
  const auto local_range = executionRange.get_local_range();

  if (global_range % local_range != range<Dimensions>{})
    throw sycl::exception{errc::nd_range,
                          "global size is not evenly divisible by local size"};

  struct Lane {
    std::queue<nd_item<Dimensions>> tasks;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    bool should_terminate = false;
  };

  std::vector<Lane> lanes(local_range.size());
  std::vector<std::thread> threads;
  threads.reserve(local_range.size());

  std::barrier work_group_barrier{
      static_cast<std::ptrdiff_t>(local_range.size())};

  auto work = [&](int index) {
    auto &lane = lanes[index];
    auto &tasks = lane.tasks;
    auto &queue_mutex = lane.queue_mutex;
    auto &mutex_condition = lane.mutex_condition;
    auto &should_terminate = lane.should_terminate;
    while (true) {
      std::unique_lock<std::mutex> lock(queue_mutex);
      while (!should_terminate && tasks.empty())
        mutex_condition.wait(
            lock, [&] { return !tasks.empty() || should_terminate; });
      if (should_terminate && tasks.empty())
        return;
      nd_item<Dimensions> task = tasks.front();
      tasks.pop();
      lock.unlock();
      launch(task);
      work_group_barrier.arrive_and_wait();
    }
  };

  for (int i = 0; i < local_range.size(); i++) {
    threads.emplace_back(std::thread{work, i});
  }

  auto launch_work_group = [&](const id<Dimensions> offset) {
    for (std::size_t idx = 0; idx < local_range.size(); idx++) {
      const id<Dimensions> local_id = detail::unlinearize(idx, local_range);
      const id<Dimensions> global_id = global_offset + offset + local_id;
      const nd_item<Dimensions> item = detail::make_nd_item(
          &work_group_barrier, global_id, local_id, executionRange);
      {
        std::unique_lock<std::mutex> lock(lanes[idx].queue_mutex);
        lanes[idx].tasks.push(item);
      }
      lanes[idx].mutex_condition.notify_one();
    }
  };

  if constexpr (Dimensions == 1) {
    for (std::size_t x = 0; x < global_range.get(0); x += local_range.get(0))
      launch_work_group(x);
  } else if constexpr (Dimensions == 2) {
    for (std::size_t x = 0; x < global_range.get(0); x += local_range.get(0))
      for (std::size_t y = 0; y < global_range.get(1); y += local_range.get(1))
        launch_work_group({x, y});
  } else if constexpr (Dimensions == 3) {
    for (std::size_t x = 0; x < global_range.get(0); x += local_range.get(0))
      for (std::size_t y = 0; y < global_range.get(1); y += local_range.get(1))
        for (std::size_t z = 0; z < global_range.get(2);
             z += local_range.get(2))
          launch_work_group({x, y, z});
  }

  for (Lane &lane : lanes) {
    {
      std::unique_lock<std::mutex> lock(lane.queue_mutex);
      lane.should_terminate = true;
    }
    lane.mutex_condition.notify_all();
  }
  for (auto &thread : threads) {
    thread.join();
  }
  lanes.clear();
}

} // namespace detail

template <typename T, typename BinaryOperation, int Dimensions> class reducer;

class handler {
private:
  // implementation defined constructor
  handler() = default;

public:
  template <typename DataT, int Dimensions, access_mode AccessMode,
            target AccessTarget, access::placeholder IsPlaceholder>
  void
  require(accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder>
              acc) {
    const void *const target = detail::get_accessor_data_pointer(acc);
    const std::size_t size = [&] {
      if constexpr (Dimensions > 0)
        return acc.get_range().size() * sizeof(DataT);
      else
        return sizeof(DataT);
    }();
    const bool no_init = acc.template has_property<property::no_init>();
    const detail::requisite::access_mode mode = ([&] {
      using namespace detail;
      if constexpr (AccessMode == access_mode::read)
        return requisite::access_mode::read;
      if constexpr (AccessMode == access_mode::write)
        return no_init ? requisite::access_mode::no_init_write
                       : requisite::access_mode::write;
      if constexpr (AccessMode == access_mode::read_write ||
                    AccessMode == access_mode::atomic)
        return no_init ? requisite::access_mode::no_init_read_write
                       : requisite::access_mode::read_write;
      if constexpr (AccessMode == access_mode::discard_write)
        return requisite::access_mode::no_init_write;
      if constexpr (AccessMode == access_mode::discard_read_write)
        return requisite::access_mode::no_init_read_write;
      return requisite::access_mode::write;
    })();
    m_task.add_requisite(detail::requisite{target, size, mode});
  }

  void depends_on(event depEvent) { m_task.m_dependencies.push_back(depEvent); }

  void depends_on(const std::vector<event> &depEvents) {
    for (const event &dep_event : depEvents)
      depends_on(dep_event);
  }

  //----- Backend interoperability interface
  //
  template <typename T> void set_arg(int argIndex, T &&arg);

  template <typename... Ts> void set_args(Ts &&...args);

  //------ Kernel dispatch API
  //
  // Note: In all kernel dispatch functions, the template parameter
  // "typename KernelName" is optional.
  //
  template <typename KernelName = detail::unnamed_kernel, typename KernelType>
  void single_task(const KernelType &kernelFunc) {
    std::ignore = detail::kernel_registry_v<KernelName, KernelType>;
    assert_launchable<KernelName>(range<1>{});
    if (m_local_accessors > 0)
      throw sycl::exception{errc::kernel_argument};
    auto specialization_constants = get_spec_constants();
    set_action([=]() {
      if constexpr (std::is_invocable_v<KernelType, kernel_handler>) {
        kernel_handler kh{specialization_constants};
        kernelFunc(kh);
      } else {
        kernelFunc();
      }
    });
  }

  // Parameter pack acts as-if: Reductions&&... reductions, const KernelType
  // &kernelFunc
  template <typename KernelName = detail::unnamed_kernel, int Dimensions,
            typename... Rest>
  void parallel_for(range<Dimensions> numWorkItems, Rest &&...rest) {
    const auto kernel =
        std::get<sizeof...(rest) - 1>(std::forward_as_tuple(rest...));
    std::ignore = detail::kernel_registry_v<KernelName, decltype(kernel)>;
    auto specialization_constants = get_spec_constants();

    set_action([=]() mutable {
      auto launch_over_range = [numWorkItems](auto launch) {
        if constexpr (Dimensions == 1) {
          const std::size_t bound_x = numWorkItems.get(0);
          for (std::size_t x = 0; x < bound_x; x++)
            launch(detail::make_item(id{x}, numWorkItems));
        } else if constexpr (Dimensions == 2) {
          const std::size_t bound_x = numWorkItems.get(0);
          const std::size_t bound_y = numWorkItems.get(1);
          for (std::size_t x = 0; x < bound_x; x++)
            for (std::size_t y = 0; y < bound_y; y++)
              launch(detail::make_item(id{x, y}, numWorkItems));
        } else if constexpr (Dimensions == 3) {
          const std::size_t bound_x = numWorkItems.get(0);
          const std::size_t bound_y = numWorkItems.get(1);
          const std::size_t bound_z = numWorkItems.get(2);
          for (std::size_t x = 0; x < bound_x; x++)
            for (std::size_t y = 0; y < bound_y; y++)
              for (std::size_t z = 0; z < bound_z; z++)
                launch(detail::make_item(id{x, y, z}, numWorkItems));
        }
      };

      auto execute_parallel_for =
          [=]<typename KernelFunc, typename... Reducers>(
              KernelFunc &&kernelFunc, Reducers &&...reducers) {
            auto launch = [&](const item<Dimensions, false> &it) {
              kernel_handler kh{specialization_constants};
              [&, ... reducer_values = reducer{reducers}] mutable {
                if constexpr (std::is_invocable_v<
                                  KernelFunc, item<Dimensions, false>,
                                  decltype(reducer_values)..., kernel_handler>)
                  kernelFunc(it, reducer_values..., kh);
                else
                  kernelFunc(it, reducer_values...);
              }();
            };
            launch_over_range(launch);
          };

      auto launch_parallel_for =
          [execute_parallel_for]<typename RestTuple,
                                 std::size_t... ReductionIndices,
                                 std::size_t KernelIndex>(
              RestTuple &&restTuple, std::index_sequence<ReductionIndices...>,
              std::index_sequence<KernelIndex>) {
            const auto &kernel_func = std::get<KernelIndex>(restTuple);
            execute_parallel_for(kernel_func,
                                 std::get<ReductionIndices>(restTuple)...);
          };

      launch_parallel_for(std::forward_as_tuple(std::forward<Rest>(rest)...),
                          std::make_index_sequence<sizeof...(Rest) - 1>{},
                          std::index_sequence<sizeof...(Rest) - 1>{});
    });
  }

  template <typename KernelName = detail::unnamed_kernel, typename... Rest>
  void parallel_for(std::size_t numWorkItems, Rest &&...rest) {
    parallel_for<KernelName, 1, Rest...>(range(numWorkItems),
                                         std::forward<Rest...>(rest)...);
  }

  template <typename KernelName = detail::unnamed_kernel, typename... Rest>
  void parallel_for(std::initializer_list<std::size_t> numWorkItems,
                    Rest &&...rest) {
    const auto kernel =
        std::get<sizeof...(rest) - 1>(std::forward_as_tuple(rest...));
    const std::vector<std::size_t> dims{numWorkItems};
    if constexpr (std::is_invocable_v<decltype(kernel), item<1, false>,
                                      kernel_handler> ||
                  std::is_invocable_v<decltype(kernel), item<1, false>>)
      parallel_for<KernelName, 1, Rest...>(range<1>{dims[0]},
                                           std::forward<Rest...>(rest)...);
    else if constexpr (std::is_invocable_v<decltype(kernel), item<2, false>,
                                           kernel_handler> ||
                       std::is_invocable_v<decltype(kernel), item<2, false>>)
      parallel_for<KernelName, 2, Rest...>(range<2>{dims[0], dims[1]},
                                           std::forward<Rest...>(rest)...);
    else if constexpr (std::is_invocable_v<decltype(kernel), item<3, false>,
                                           kernel_handler> ||
                       std::is_invocable_v<decltype(kernel), item<3, false>>)
      parallel_for<KernelName, 3, Rest...>(range<3>{dims[0], dims[1], dims[2]},
                                           std::forward<Rest...>(rest)...);
  }

  // Deprecated in SYCL 2020.
  template <typename KernelName = detail::unnamed_kernel, typename KernelType,
            int Dimensions>
  void parallel_for(range<Dimensions> numWorkItems,
                    id<Dimensions> workItemOffset, KernelType &&kernelFunc) {
    std::ignore = detail::kernel_registry_v<KernelName, KernelType>;
    auto specialization_constants = get_spec_constants();
    set_action([=]() {
      auto launch = [&](const item<Dimensions, true> &it) {
        if constexpr (std::is_invocable_v<decltype(kernelFunc),
                                          item<Dimensions, false>,
                                          kernel_handler>) {
          kernel_handler kh{specialization_constants};
          kernelFunc(it, kh);
        } else {
          kernelFunc(it);
        }
      };

      if constexpr (Dimensions == 1) {
        std::size_t bound_x = numWorkItems.get(0);
        for (std::size_t x = 0; x < bound_x; x++)
          launch(detail::make_item(workItemOffset + id{x}, numWorkItems,
                                   workItemOffset));
      } else if constexpr (Dimensions == 2) {
        std::size_t bound_x = numWorkItems.get(0);
        std::size_t bound_y = numWorkItems.get(1);
        for (std::size_t x = 0; x < bound_x; x++)
          for (std::size_t y = 0; y < bound_y; y++)
            launch(detail::make_item(workItemOffset + id{x, y}, numWorkItems,
                                     workItemOffset));
      } else if constexpr (Dimensions == 3) {
        std::size_t bound_x = numWorkItems.get(0);
        std::size_t bound_y = numWorkItems.get(1);
        std::size_t bound_z = numWorkItems.get(2);
        for (std::size_t x = 0; x < bound_x; x++)
          for (std::size_t y = 0; y < bound_y; y++)
            for (std::size_t z = 0; z < bound_z; z++)
              launch(detail::make_item(workItemOffset + id{x, y, z},
                                       numWorkItems, workItemOffset));
      }
    });
  }

  // Parameter pack acts as-if: Reductions&&... reductions, const KernelType
  // &kernelFunc
  template <typename KernelName = detail::unnamed_kernel, int Dimensions,
            typename... Rest>
  void parallel_for(nd_range<Dimensions> executionRange, Rest &&...rest) {
    const auto kernel =
        std::get<sizeof...(rest) - 1>(std::forward_as_tuple(rest...));
    std::ignore = detail::kernel_registry_v<KernelName, decltype(kernel)>;
    auto specialization_constants = get_spec_constants();
    assert_launchable<KernelName>(executionRange.get_local_range());

    set_action([=]() mutable {
      auto execute_parallel_for =
          [=]<typename KernelFunc, typename... Reducers>(
              KernelFunc &&kernelFunc, Reducers &&...reducers) {
            auto launch = [&](const nd_item<Dimensions> &it) {
              kernel_handler kh{specialization_constants};
              [&, ... reducer_values = reducer{reducers}] mutable {
                if constexpr (std::is_invocable_v<
                                  KernelFunc, nd_item<Dimensions>,
                                  decltype(reducer_values)..., kernel_handler>)
                  kernelFunc(it, reducer_values..., kh);
                else
                  kernelFunc(it, reducer_values...);
              }();
            };
            detail::nd_launch(executionRange, launch);
          };

      auto launch_parallel_for =
          [execute_parallel_for]<typename RestTuple,
                                 std::size_t... ReductionIndices,
                                 std::size_t KernelIndex>(
              RestTuple &&restTuple, std::index_sequence<ReductionIndices...>,
              std::index_sequence<KernelIndex>) {
            const auto &kernel_func = std::get<KernelIndex>(restTuple);
            execute_parallel_for(kernel_func,
                                 std::get<ReductionIndices>(restTuple)...);
          };

      launch_parallel_for(std::forward_as_tuple(std::forward<Rest>(rest)...),
                          std::make_index_sequence<sizeof...(Rest) - 1>{},
                          std::index_sequence<sizeof...(Rest) - 1>{});
    });
  }

  template <typename KernelName = detail::unnamed_kernel,
            typename WorkgroupFunctionType, int Dimensions>
  void parallel_for_work_group(range<Dimensions> numWorkGroups,
                               const WorkgroupFunctionType &kernelFunc) {
    parallel_for_work_group(numWorkGroups, range<Dimensions>{} + 1, kernelFunc);
  }

  template <typename KernelName = detail::unnamed_kernel,
            typename WorkgroupFunctionType, int Dimensions>
  void parallel_for_work_group(range<Dimensions> numWorkGroups,
                               range<Dimensions> workGroupSize,
                               const WorkgroupFunctionType &kernelFunc) {
    auto specialization_constants = get_spec_constants();
    std::ignore = detail::kernel_registry_v<KernelName, WorkgroupFunctionType>;
    assert_launchable<KernelName>(workGroupSize);

    set_action([=]() {
      const auto global_range = numWorkGroups * workGroupSize;
      const auto local_range = workGroupSize;
      const auto execution_range = nd_range{global_range, local_range};

      const auto launch = [&](const id<Dimensions> offset) {
        const id<Dimensions> local_id{};
        const id<Dimensions> global_id = offset + local_id;
        const nd_item<Dimensions> item =
            detail::make_nd_item(nullptr, global_id, local_id, execution_range);

        group<Dimensions> g = item.get_group();
        if constexpr (std::is_invocable_v<WorkgroupFunctionType,
                                          group<Dimensions>, kernel_handler>) {
          kernel_handler kh{specialization_constants};
          kernelFunc(g, kh);
        } else {
          kernelFunc(g);
        }
      };

      if constexpr (Dimensions == 1) {
        for (std::size_t x = 0; x < global_range.get(0);
             x += local_range.get(0))
          launch(x);
      } else if constexpr (Dimensions == 2) {
        for (std::size_t x = 0; x < global_range.get(0);
             x += local_range.get(0))
          for (std::size_t y = 0; y < global_range.get(1);
               y += local_range.get(1))
            launch({x, y});
      } else if constexpr (Dimensions == 3) {
        for (std::size_t x = 0; x < global_range.get(0);
             x += local_range.get(0))
          for (std::size_t y = 0; y < global_range.get(1);
               y += local_range.get(1))
            for (std::size_t z = 0; z < global_range.get(2);
                 z += local_range.get(2))
              launch({x, y, z});
      }
    });
  }

  void single_task(const kernel &kernelObject);

  template <int Dimensions>
  void parallel_for(range<Dimensions> numWorkItems, const kernel &kernelObject);

  template <int Dimensions>
  void parallel_for(nd_range<Dimensions> ndRange, const kernel &kernelObject);

  //------ USM functions
  //

  void memcpy(void *dest, const void *src, std::size_t numBytes) {
    set_action([=]() { std::memcpy(dest, src, numBytes); });
  }

  template <typename T> void copy(const T *src, T *dest, std::size_t count) {
    set_action([=]() { std::copy(src, src + count, dest); });
  }

  void memset(void *ptr, int value, std::size_t numBytes) {
    set_action([=]() { std::memset(ptr, value, numBytes); });
  }

  template <typename T>
  void fill(void *ptr, const T &pattern, std::size_t count) {
    set_action([=]() {
      std::fill(static_cast<T *>(ptr), static_cast<T *>(ptr) + count, pattern);
    });
  }

  void prefetch(const void *ptr, std::size_t numBytes) {
    std::ignore = ptr;
    std::ignore = numBytes;
  }

  void mem_advise(const void *ptr, std::size_t numBytes, int advice) {
    std::ignore = ptr;
    std::ignore = numBytes;
    std::ignore = advice;
  }

  //------ Explicit memory operation APIs
  //
  template <typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt,
            access::placeholder IsPlaceholder, typename DestT>
  void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, IsPlaceholder> src,
            std::shared_ptr<DestT> dest) {
    set_action([=]() {
      for (std::size_t i = 0; i < src.get_range().size(); i++)
        dest.get()[i] = src[detail::unlinearize(i, src.get_range())];
    });
  }

  template <typename SrcT, typename DestT, int DestDim, access_mode DestMode,
            target DestTgt, access::placeholder IsPlaceholder>
  void copy(std::shared_ptr<SrcT> src,
            accessor<DestT, DestDim, DestMode, DestTgt, IsPlaceholder> dest) {
    set_action([=]() {
      for (std::size_t i = 0; i < dest.get_range().size(); i++)
        dest[detail::unlinearize(i, dest.get_range())] = src.get()[i];
    });
  }

  template <typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt,
            access::placeholder IsPlaceholder, typename DestT>
  void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, IsPlaceholder> src,
            DestT *dest) {
    set_action([=]() {
      for (std::size_t i = 0; i < src.get_range().size(); i++)
        dest[i] = src[detail::unlinearize(i, src.get_range())];
    });
  }

  template <typename SrcT, typename DestT, int DestDim, access_mode DestMode,
            target DestTgt, access::placeholder IsPlaceholder>
  void copy(const SrcT *src,
            accessor<DestT, DestDim, DestMode, DestTgt, IsPlaceholder> dest) {
    set_action([=]() {
      for (std::size_t i = 0; i < dest.get_range().size(); i++)
        dest[detail::unlinearize(i, dest.get_range())] = src[i];
    });
  }

  template <typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt,
            access::placeholder SrcIsPlaceholder, typename DestT, int DestDim,
            access_mode DestMode, target DestTgt,
            access::placeholder DestIsPlaceholder>
  void
  copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, SrcIsPlaceholder> src,
       accessor<DestT, DestDim, DestMode, DestTgt, DestIsPlaceholder> dest) {
    if (src.get_range().size() > dest.get_range().size())
      throw sycl::exception{errc::invalid};
    set_action([=]() {
      for (std::size_t i = 0; i < src.get_range().size(); i++)
        dest[detail::unlinearize(i, dest.get_range())] =
            src[detail::unlinearize(i, src.get_range())];
    });
  }

  template <typename T, int Dim, access_mode Mode, target Tgt,
            access::placeholder IsPlaceholder>
  void update_host(accessor<T, Dim, Mode, Tgt, IsPlaceholder> acc) {
    std::ignore = acc;
  }

  template <typename T, int Dim, access_mode Mode, target Tgt,
            access::placeholder IsPlaceholder>
  void fill(accessor<T, Dim, Mode, Tgt, IsPlaceholder> dest, const T &src) {
    set_action([=]() { std::fill(dest.begin(), dest.end(), src); });
  }

  void
  use_kernel_bundle(const kernel_bundle<bundle_state::executable> &execBundle) {
    if (!m_specializationConstants.empty())
      throw sycl::exception{errc::invalid,
                            "Cannot call use_kernel_bundle after setting "
                            "specialization constants"};
    m_usedKernelBundle = execBundle;
    m_specializationConstants = execBundle.m_impl->m_specializationConstants;
  }

  template <auto &SpecName>
  void set_specialization_constant(
      typename std::remove_reference_t<decltype(SpecName)>::value_type value) {
    if (m_usedKernelBundle)
      throw sycl::exception{errc::invalid,
                            "Cannot set specialization constant when using "
                            "kernel bundle in handler"};
    const void *index = static_cast<const void *>(&SpecName);
    m_specializationConstants[index] = value;
  }

  template <auto &SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::value_type
  get_specialization_constant() {
    if (m_usedKernelBundle)
      throw sycl::exception{errc::invalid,
                            "Cannot get specialization constant when using "
                            "kernel bundle in handler"};
    const void *index = static_cast<const void *>(&SpecName);
    return m_specializationConstants.contains(index)
               ? std::any_cast<typename std::remove_reference_t<
                     decltype(SpecName)>::value_type>(
                     m_specializationConstants[index])
               : detail::get_specialization_default_value(SpecName);
  }

  template <typename T> void host_task(T &&hostTaskCallable) {
    set_action([=]() {
      using interop_handle = int;
      if constexpr (std::is_invocable_v<T, interop_handle>)
        hostTaskCallable(0);
      else
        hostTaskCallable();
    });
  }

private:
  friend class queue;
  friend std::any get_constant_value(const handler *, const void *);

  template <typename DataT, int Dimensions>
  friend void
  detail::registerAccessor(handler &,
                           const local_accessor<DataT, Dimensions> &);

  void set_action(std::function<void()> action) {
    m_task.m_action = std::move(action);
  }

  std::shared_ptr<constant_map> get_spec_constants() const {
    return std::make_shared<constant_map>(
        m_usedKernelBundle
            ? m_usedKernelBundle->m_impl->m_specializationConstants
            : m_specializationConstants);
  }

  template <typename KernelName, int Dimensions>
  void assert_launchable(range<Dimensions> workGroupSize) {
    const auto kernel_id = get_kernel_id<KernelName>();
    if (m_usedKernelBundle)
      if (!m_usedKernelBundle->has_kernel<KernelName>())
        throw sycl::exception{errc::kernel_not_supported,
                              "Kernel not found in used kernel bundle"};
    const bool compatible = is_compatible({kernel_id}, device{});
    if (!compatible)
      throw sycl::exception{errc::kernel_not_supported,
                            "Kernel is not compatible with device"};

    const auto &attributes = detail::get_kernel_attributes(kernel_id);
    for (const auto &attr : attributes) {
      using namespace detail;
      if (std::holds_alternative<kernel_attributes::reqd_work_group_size_1d>(
              attr)) {
        const auto &reqd =
            std::get<kernel_attributes::reqd_work_group_size_1d>(attr);
        if constexpr (Dimensions == 1)
          if (workGroupSize[0] == reqd.x)
            continue;
        throw sycl::exception{errc::nd_range,
                              "Kernel requires different work-group size"};
      } else if (std::holds_alternative<
                     kernel_attributes::reqd_work_group_size_2d>(attr)) {
        const auto &reqd =
            std::get<kernel_attributes::reqd_work_group_size_2d>(attr);
        if constexpr (Dimensions == 2)
          if (workGroupSize[0] == reqd.x && workGroupSize[1] == reqd.y)
            continue;
        throw sycl::exception{errc::nd_range,
                              "Kernel requires different work-group size"};
      } else if (std::holds_alternative<
                     kernel_attributes::reqd_work_group_size_3d>(attr)) {
        const auto &reqd =
            std::get<kernel_attributes::reqd_work_group_size_3d>(attr);
        if constexpr (Dimensions == 3)
          if (workGroupSize[0] == reqd.x && workGroupSize[1] == reqd.y &&
              workGroupSize[2] == reqd.z)
            continue;
        throw sycl::exception{errc::nd_range,
                              "Kernel requires different work-group size"};
      }
    }
  }

  int m_local_accessors = 0;
  detail::task m_task;
  std::unordered_map<const void *, std::any> m_specializationConstants;
  std::optional<kernel_bundle<bundle_state::executable>> m_usedKernelBundle;
};

template <typename DataT, int Dimensions>
void detail::registerAccessor(handler &cgh,
                              const local_accessor<DataT, Dimensions> &) {
  cgh.m_local_accessors++;
}

template <typename DataT, int Dimensions, access_mode AccessMode,
          target AccessTarget, access::placeholder IsPlaceholder>
void register_accessor(handler &cgh,
                       const accessor<DataT, Dimensions, AccessMode,
                                      AccessTarget, IsPlaceholder> &acc) {
  cgh.require(acc);
}

} // namespace sycl
