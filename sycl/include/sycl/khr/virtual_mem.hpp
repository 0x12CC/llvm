#pragma once

#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include <sycl/khr/properties.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace khr {

enum class granularity_mode { minimum, recommended };
enum class address_access_mode { none, read, read_write };

namespace virtual_mem_detail {

inline bool virtual_mem_device_in_context(const device &SyclDevice,
                                          const context &SyclContext) {
  const auto Devices = SyclContext.get_devices();
  device Current = SyclDevice;
  while (true) {
    if (std::find(Devices.begin(), Devices.end(), Current) != Devices.end())
      return true;
    try {
      Current = Current.get_info<sycl::info::device::parent_device>();
    } catch (const sycl::exception &Error) {
      if (Error.code() != errc::invalid)
        throw;
      return false;
    }
  }
}

inline void virtual_mem_check_device(const device &SyclDevice,
                                     const context &SyclContext) {
  if (!virtual_mem_device_in_context(SyclDevice, SyclContext))
    throw sycl::exception(make_error_code(errc::invalid),
                          "Device is not associated with the context.");
}

inline void virtual_mem_check_size(std::size_t NumBytes) {
  if (NumBytes == 0)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Virtual memory size must be nonzero.");
}

inline ext::oneapi::experimental::granularity_mode
virtual_mem_granularity_mode(granularity_mode Mode) {
  return Mode == granularity_mode::minimum
             ? ext::oneapi::experimental::granularity_mode::minimum
             : ext::oneapi::experimental::granularity_mode::recommended;
}

inline ext::oneapi::experimental::address_access_mode
virtual_mem_access_mode(address_access_mode Mode) {
  switch (Mode) {
  case address_access_mode::none:
    return ext::oneapi::experimental::address_access_mode::none;
  case address_access_mode::read:
    return ext::oneapi::experimental::address_access_mode::read;
  case address_access_mode::read_write:
    return ext::oneapi::experimental::address_access_mode::read_write;
  }
  throw sycl::exception(make_error_code(errc::invalid), "Invalid access mode.");
}

} // namespace virtual_mem_detail

inline std::size_t
get_mem_granularity(const device &SyclDevice, const context &SyclContext,
                    granularity_mode Mode = granularity_mode::recommended) {
  virtual_mem_detail::virtual_mem_check_device(SyclDevice, SyclContext);
  return ext::oneapi::experimental::get_mem_granularity(
      SyclDevice, SyclContext,
      virtual_mem_detail::virtual_mem_granularity_mode(Mode));
}

inline std::size_t
get_mem_granularity(const context &SyclContext,
                    granularity_mode Mode = granularity_mode::recommended) {
  return ext::oneapi::experimental::get_mem_granularity(
      SyclContext, virtual_mem_detail::virtual_mem_granularity_mode(Mode));
}

inline std::uintptr_t reserve_virtual_mem(std::uintptr_t Start,
                                          std::size_t NumBytes,
                                          const context &SyclContext) {
  virtual_mem_detail::virtual_mem_check_size(NumBytes);
  for (const device &Dev : SyclContext.get_devices())
    if (!Dev.has(aspect::ext_oneapi_virtual_mem))
      throw sycl::exception(make_error_code(errc::feature_not_supported),
                            "Device does not support virtual memory.");
  try {
    return ext::oneapi::experimental::reserve_virtual_mem(Start, NumBytes,
                                                          SyclContext);
  } catch (const sycl::exception &) {
    if (Start != 0) {
      try {
        return ext::oneapi::experimental::reserve_virtual_mem(0, NumBytes,
                                                              SyclContext);
      } catch (const sycl::exception &) {
      }
    }
    throw sycl::exception(make_error_code(errc::invalid),
                          "Unable to reserve virtual memory.");
  }
}

inline std::uintptr_t reserve_virtual_mem(std::size_t NumBytes,
                                          const context &SyclContext) {
  return khr::reserve_virtual_mem(0, NumBytes, SyclContext);
}

inline void free_virtual_mem(std::uintptr_t Ptr, std::size_t NumBytes,
                             const context &SyclContext) {
  ext::oneapi::experimental::free_virtual_mem(Ptr, NumBytes, SyclContext);
}

class physical_mem {
  using impl_type = ext::oneapi::experimental::physical_mem;
  impl_type Impl;

  static impl_type create(const device &SyclDevice, const context &SyclContext,
                          std::size_t NumBytes) {
    virtual_mem_detail::virtual_mem_check_size(NumBytes);
    virtual_mem_detail::virtual_mem_check_device(SyclDevice, SyclContext);
    return impl_type(SyclDevice, SyclContext, NumBytes);
  }

public:
  template <
      typename PropertyOrList = empty_properties_t,
      std::enable_if_t<is_property_or_list_for_v<PropertyOrList, physical_mem>,
                       int> = 0>
  physical_mem(const device &SyclDevice, const context &SyclContext,
               std::size_t NumBytes, PropertyOrList = {})
      : Impl(create(SyclDevice, SyclContext, NumBytes)) {}

  template <
      typename PropertyOrList = empty_properties_t,
      std::enable_if_t<is_property_or_list_for_v<PropertyOrList, physical_mem>,
                       int> = 0>
  physical_mem(const queue &SyclQueue, std::size_t NumBytes,
               PropertyOrList Props = {})
      : physical_mem(SyclQueue.get_device(), SyclQueue.get_context(), NumBytes,
                     Props) {}

  void *map(std::uintptr_t Ptr, std::size_t NumBytes, address_access_mode Mode,
            std::size_t Offset = 0) const {
    virtual_mem_detail::virtual_mem_check_size(NumBytes);
    return Impl.map(Ptr, NumBytes,
                    virtual_mem_detail::virtual_mem_access_mode(Mode), Offset);
  }

  context get_context() const { return Impl.get_context(); }
  device get_device() const { return Impl.get_device(); }
  std::size_t size() const noexcept { return Impl.size(); }
  bool operator==(const physical_mem &Other) const {
    return Impl == Other.Impl;
  }
  bool operator!=(const physical_mem &Other) const { return !(*this == Other); }

  friend struct std::hash<physical_mem>;
};

inline void set_access_mode(const void *Ptr, std::size_t NumBytes,
                            address_access_mode Mode,
                            const context &SyclContext) {
  ext::oneapi::experimental::set_access_mode(
      Ptr, NumBytes, virtual_mem_detail::virtual_mem_access_mode(Mode),
      SyclContext);
}

inline address_access_mode get_access_mode(const void *Ptr,
                                           std::size_t NumBytes,
                                           const context &SyclContext) {
  switch (
      ext::oneapi::experimental::get_access_mode(Ptr, NumBytes, SyclContext)) {
  case ext::oneapi::experimental::address_access_mode::none:
    return address_access_mode::none;
  case ext::oneapi::experimental::address_access_mode::read:
    return address_access_mode::read;
  case ext::oneapi::experimental::address_access_mode::read_write:
    return address_access_mode::read_write;
  }
  throw sycl::exception(make_error_code(errc::invalid), "Invalid access mode.");
}

inline void unmap(const void *Ptr, std::size_t NumBytes,
                  const context &SyclContext) {
  ext::oneapi::experimental::unmap(Ptr, NumBytes, SyclContext);
}

} // namespace khr
} // namespace _V1
} // namespace sycl

template <> struct std::hash<sycl::khr::physical_mem> {
  std::size_t operator()(const sycl::khr::physical_mem &Memory) const noexcept {
    return std::hash<sycl::ext::oneapi::experimental::physical_mem>{}(
        Memory.Impl);
  }
};

#endif
