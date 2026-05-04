#pragma once

#include "function_objects.hpp"
#include "handler.hpp"
#include "property.hpp"
#include "property_list.hpp"

namespace sycl::detail {

template <typename T, typename BinaryOperation> struct reducer_impl {
  std::mutex m_mutex;
  T *m_value;
  BinaryOperation m_combiner;
  std::optional<T> m_identity;
  property_list m_props;
};

template <typename T, typename BinaryOperation> struct reducer_span_impl {
  std::vector<std::shared_ptr<reducer_impl<T, BinaryOperation>>> m_reducers;
  BinaryOperation m_combiner;
  std::optional<T> m_identity;
  property_list m_props;
};

} // namespace sycl::detail

namespace sycl {

namespace property::reduction {

struct initialize_to_identity {};

} // namespace property::reduction

template <>
struct is_property<property::reduction::initialize_to_identity>
    : std::true_type {};

template <typename T, typename BinaryOperation, int Dimensions> class reducer;

template <typename BufferT, typename BinaryOperation>
auto reduction(BufferT, handler &, BinaryOperation, const property_list & = {});

template <typename T, typename BinaryOperation>
auto reduction(T *, BinaryOperation, const property_list & = {});

template <typename T, std::size_t Extent, typename BinaryOperation>
auto reduction(std::span<T, Extent>, BinaryOperation,
               const property_list & = {});

template <typename BufferT, typename BinaryOperation>
auto reduction(BufferT, handler &, const typename BufferT::value_type &,
               BinaryOperation, const property_list & = {});

template <typename T, typename BinaryOperation>
auto reduction(T *, const T &, BinaryOperation, const property_list & = {});

template <typename T, std::size_t Extent, typename BinaryOperation>
auto reduction(std::span<T, Extent>, const T &, BinaryOperation,
               const property_list & = {});

template <typename T, typename BinaryOperation>
class reducer<T, BinaryOperation, 0> {
public:
  using value_type = T;
  using binary_operation = BinaryOperation;
  static constexpr int dimensions = 0;

  reducer(const reducer &) = delete;
  reducer(reducer &&) = delete;
  reducer &operator=(const reducer &) = delete;
  reducer &operator=(reducer &&) = delete;

  reducer &combine(const T &partial)
    requires(dimensions == 0)
  {
    std::scoped_lock lock{m_impl->m_mutex};
    *m_impl->m_value = m_impl->m_combiner(*m_impl->m_value, partial);
    return *this;
  }

  template <typename Property> bool has_property() const noexcept {
    return detail::has_property<Property>(m_impl->m_props);
  }

  template <typename Property> Property get_property() const {
    return detail::get_property<Property>(m_impl->m_props);
  }

  /* Only available if identity value is known */
  T identity() const { return m_impl->m_identity.value(); }

  friend reducer &operator+=(reducer &r, const T &v)
    requires(dimensions == 0 && (std::is_same_v<BinaryOperation, std::plus<>> ||
                                 std::is_same_v<BinaryOperation, std::plus<T>>))
  {
    return r.combine(v);
  }

  friend reducer &operator*=(reducer &r, const T &v)
    requires(dimensions == 0 &&
             (std::is_same_v<BinaryOperation, std::multiplies<>> ||
              std::is_same_v<BinaryOperation, std::multiplies<T>>))
  {
    return r.combine(v);
  }

  friend reducer &operator&=(reducer &r, const T &v)
    requires(dimensions == 0 && std::is_integral_v<T> &&
             (std::is_same_v<BinaryOperation, std::bit_and<>> ||
              std::is_same_v<BinaryOperation, std::bit_and<T>>))
  {
    return r.combine(v);
  }

  friend reducer &operator|=(reducer &r, const T &v)
    requires(dimensions == 0 && std::is_integral_v<T> &&
             (std::is_same_v<BinaryOperation, std::bit_or<>> ||
              std::is_same_v<BinaryOperation, std::bit_or<T>>))
  {
    return r.combine(v);
  }

  friend reducer &operator^=(reducer &r, const T &v)
    requires(dimensions == 0 && std::is_integral_v<T> &&
             (std::is_same_v<BinaryOperation, std::bit_xor<>> ||
              std::is_same_v<BinaryOperation, std::bit_xor<T>>))
  {
    return r.combine(v);
  }

  friend reducer &operator++(reducer &r)
    requires(dimensions == 0 && std::is_integral_v<T> &&
             !std::is_same_v<T, bool> &&
             (std::is_same_v<BinaryOperation, std::plus<>> ||
              std::is_same_v<BinaryOperation, std::plus<T>>))
  {
    return r.combine(T{1});
  }

private:
  template <typename BufferT, typename BinaryOperation_>
  friend auto reduction(BufferT, handler &, BinaryOperation_,
                        const property_list &);

  template <typename T_, typename BinaryOperation_>
  friend auto reduction(T_ *, BinaryOperation_, const property_list &);

  template <typename BufferT, typename BinaryOperation_>
  friend auto reduction(BufferT, handler &,
                        const typename BufferT::value_type &, BinaryOperation_,
                        const property_list &);

  template <typename T_, typename BinaryOperation_>
  friend auto reduction(T_ *, const T_ &, BinaryOperation_,
                        const property_list &);

  friend class reducer<T, BinaryOperation, 1>;

  friend handler;

  reducer(std::shared_ptr<detail::reducer_impl<T, BinaryOperation>> impl)
      : m_impl{impl} {}

  reducer(T *value, BinaryOperation combiner, const property_list &props) {
    if constexpr (has_known_identity_v<BinaryOperation, T>)
      m_impl->m_identity = known_identity_v<BinaryOperation, T>;

    m_impl->m_value = value;
    m_impl->m_combiner = combiner;
    m_impl->m_props = props;

    if (has_property<property::reduction::initialize_to_identity>())
      *value = m_impl->m_identity.value();
  }

  reducer(T *value, BinaryOperation combiner, const T &identity,
          const property_list &props) {
    m_impl->m_value = value;
    m_impl->m_combiner = combiner;
    m_impl->m_identity = identity;
    m_impl->m_props = props;

    if (has_property<property::reduction::initialize_to_identity>())
      *value = m_impl->m_identity.value();
  }

  std::shared_ptr<detail::reducer_impl<T, BinaryOperation>> m_impl{
      std::make_shared<detail::reducer_impl<T, BinaryOperation>>()};
};

template <typename T, typename BinaryOperation>
class reducer<T, BinaryOperation, 1> {
public:
  using value_type = T;
  using binary_operation = BinaryOperation;
  static constexpr int dimensions = 1;

  reducer(const reducer &) = delete;
  reducer(reducer &&) = delete;
  reducer &operator=(const reducer &) = delete;
  reducer &operator=(reducer &&) = delete;

  reducer<T, BinaryOperation, dimensions - 1> &operator[](std::size_t index)
    requires(dimensions > 0)
  {
    return *m_reducers[index];
  }

  template <typename Property> bool has_property() const noexcept {
    return detail::has_property<Property>(m_impl->m_props);
  }

  template <typename Property> Property get_property() const {
    return detail::get_property<Property>(m_impl->m_props);
  }

  /* Only available if identity value is known */
  T identity() const { return m_impl->m_identity.value(); }

private:
  template <typename T_, std::size_t Extent, typename BinaryOperation_>
  friend auto reduction(std::span<T_, Extent>, BinaryOperation_,
                        const property_list &);

  template <typename T_, std::size_t Extent, typename BinaryOperation_>
  friend auto reduction(std::span<T_, Extent>, const T_ &, BinaryOperation_,
                        const property_list &);

  friend handler;

  ~reducer() {
    for (auto r : m_reducers)
      delete r;
  }

  reducer(std::shared_ptr<detail::reducer_span_impl<T, BinaryOperation>> impl)
      : m_impl{impl} {
    m_reducers.reserve(impl->m_reducers.size());
    for (const auto &reducer_impl : impl->m_reducers)
      m_reducers.push_back(new reducer<T, BinaryOperation, 0>(reducer_impl));
  }

  reducer(std::span<T, std::dynamic_extent> values, BinaryOperation combiner,
          const property_list &props) {
    if constexpr (has_known_identity_v<BinaryOperation, T>)
      m_impl->m_identity = known_identity_v<BinaryOperation, T>;

    m_impl->m_combiner = combiner;
    m_impl->m_props = props;

    for (int i = 0; i < values.size(); ++i) {
      auto reducer_impl =
          std::make_shared<detail::reducer_impl<T, BinaryOperation>>();
      reducer_impl->m_value = &values[i];
      reducer_impl->m_combiner = combiner;
      reducer_impl->m_props = props;

      if (has_property<property::reduction::initialize_to_identity>())
        *reducer_impl->m_value = m_impl->m_identity.value();

      m_impl->m_reducers.push_back(reducer_impl);
      m_reducers.push_back(new reducer<T, BinaryOperation, 0>(reducer_impl));
    }
  }

  reducer(std::span<T, std::dynamic_extent> values, BinaryOperation combiner,
          const T &identity, const property_list &props) {
    m_impl->m_identity = identity;
    m_impl->m_combiner = combiner;
    m_impl->m_props = props;

    for (int i = 0; i < values.size(); ++i) {
      auto reducer_impl =
          std::make_shared<detail::reducer_impl<T, BinaryOperation>>();
      reducer_impl->m_value = &values[i];
      reducer_impl->m_combiner = combiner;
      reducer_impl->m_props = props;

      if (has_property<property::reduction::initialize_to_identity>())
        *reducer_impl->m_value = m_impl->m_identity.value();

      m_impl->m_reducers.push_back(reducer_impl);
      m_reducers.push_back(new reducer<T, BinaryOperation, 0>(reducer_impl));
    }
  }

  std::vector<reducer<T, BinaryOperation, 0> *> m_reducers;
  std::shared_ptr<detail::reducer_span_impl<T, BinaryOperation>> m_impl{
      std::make_shared<detail::reducer_span_impl<T, BinaryOperation>>()};
};

template <typename BufferT, typename BinaryOperation>
auto reduction(BufferT vars, handler &cgh, BinaryOperation combiner,
               const property_list &propList) {
  if (vars.get_range().size() != 1)
    throw sycl::exception{sycl::errc::invalid, "buffer range is not 1"};

  auto ptr = detail::get_buffer_data(vars);
  accessor acc{vars, cgh, sycl::access_mode::read_write};
  return reducer<typename BufferT::value_type, BinaryOperation, 0>{
      ptr, combiner, propList}
      .m_impl;
}

template <typename T, typename BinaryOperation>
auto reduction(T *var, BinaryOperation combiner,
               const property_list &propList) {
  return reducer<T, BinaryOperation, 0>{var, combiner, propList}.m_impl;
}

template <typename T, std::size_t Extent, typename BinaryOperation>
auto reduction(std::span<T, Extent> vars, BinaryOperation combiner,
               const property_list &propList) {
  return reducer<T, BinaryOperation, 1>{
      std::span<T, std::dynamic_extent>{vars.data(), vars.size()}, combiner,
      propList}
      .m_impl;
}

template <typename BufferT, typename BinaryOperation>
auto reduction(BufferT vars, handler &cgh,
               const typename BufferT::value_type &identity,
               BinaryOperation combiner, const property_list &propList) {
  auto ptr = detail::get_buffer_data(vars);
  accessor acc{vars, cgh, sycl::access_mode::read_write};
  return reducer<typename BufferT::value_type, BinaryOperation, 0>{
      ptr, combiner, identity, propList}
      .m_impl;
}

template <typename T, typename BinaryOperation>
auto reduction(T *var, const T &identity, BinaryOperation combiner,
               const property_list &propList) {
  return reducer<T, BinaryOperation, 0>{var, combiner, identity, propList}
      .m_impl;
}

template <typename T, std::size_t Extent, typename BinaryOperation>
auto reduction(std::span<T, Extent> vars, const T &identity,
               BinaryOperation combiner, const property_list &propList) {
  return reducer<T, BinaryOperation, 1>{
      std::span<T, std::dynamic_extent>{vars.data(), vars.size()}, combiner,
      identity, propList}
      .m_impl;
}

template <typename T, typename BinaryOperation>
reducer(std::shared_ptr<detail::reducer_impl<T, BinaryOperation>>)
    -> reducer<T, BinaryOperation, 0>;

template <typename T, typename BinaryOperation>
reducer(std::shared_ptr<detail::reducer_span_impl<T, BinaryOperation>>)
    -> reducer<T, BinaryOperation, 1>;

} // namespace sycl
