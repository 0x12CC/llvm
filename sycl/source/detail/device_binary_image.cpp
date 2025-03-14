//==----- device_binary_image.cpp --- SYCL device binary image abstraction -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_binary_image.hpp>
#include <sycl/detail/ur.hpp>

// For device image compression.
#include <detail/compression.hpp>

#include <algorithm>
#include <cstring>
#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

std::ostream &operator<<(std::ostream &Out, const DeviceBinaryProperty &P) {
  switch (P.Prop->Type) {
  case SYCL_PROPERTY_TYPE_UINT32:
    Out << "[UINT32] ";
    break;
  case SYCL_PROPERTY_TYPE_BYTE_ARRAY:
    Out << "[Byte array] ";
    break;
  case SYCL_PROPERTY_TYPE_STRING:
    Out << "[String] ";
    break;
  default:
    assert(false && "unsupported property");
    return Out;
  }
  Out << P.Prop->Name << "=";

  switch (P.Prop->Type) {
  case SYCL_PROPERTY_TYPE_UINT32:
    Out << P.asUint32();
    break;
  case SYCL_PROPERTY_TYPE_BYTE_ARRAY: {
    ByteArray BA = P.asByteArray();
    std::ios_base::fmtflags FlagsBackup = Out.flags();
    Out << std::hex;
    for (const auto &Byte : BA) {
      Out << "0x" << static_cast<unsigned>(Byte) << " ";
    }
    Out.flags(FlagsBackup);
    break;
  }
  case SYCL_PROPERTY_TYPE_STRING:
    Out << P.asCString();
    break;
  default:
    assert(false && "Unsupported property");
    return Out;
  }
  return Out;
}

uint32_t DeviceBinaryProperty::asUint32() const {
  assert(Prop->Type == SYCL_PROPERTY_TYPE_UINT32 && "property type mismatch");
  // if type fits into the ValSize - it is used to store the property value
  assert(Prop->ValAddr == nullptr && "primitive types must be stored inline");
  const auto *P = reinterpret_cast<const unsigned char *>(&Prop->ValSize);
  return (*P) | (*(P + 1) << 8) | (*(P + 2) << 16) | (*(P + 3) << 24);
}

ByteArray DeviceBinaryProperty::asByteArray() const {
  assert(Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY &&
         "property type mismatch");
  assert(Prop->ValSize > 0 && "property size mismatch");
  const auto *Data = ur::cast<const std::uint8_t *>(Prop->ValAddr);
  return {Data, Prop->ValSize};
}

const char *DeviceBinaryProperty::asCString() const {
  assert((Prop->Type == SYCL_PROPERTY_TYPE_STRING ||
          Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY) &&
         "property type mismatch");
  assert(Prop->ValSize > 0 && "property size mismatch");
  // Byte array stores its size in first 8 bytes
  size_t Shift = Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY ? 8 : 0;
  return ur::cast<const char *>(Prop->ValAddr) + Shift;
}

void RTDeviceBinaryImage::PropertyRange::init(sycl_device_binary Bin,
                                              const char *PropSetName) {
  assert(!this->Begin && !this->End && "already initialized");
  sycl_device_binary_property_set PS = nullptr;

  for (PS = Bin->PropertySetsBegin; PS != Bin->PropertySetsEnd; ++PS) {
    assert(PS->Name && "nameless property set - bug in the offload wrapper?");
    if (!strcmp(PropSetName, PS->Name))
      break;
  }
  if (PS == Bin->PropertySetsEnd) {
    Begin = End = nullptr;
    return;
  }
  Begin = PS->PropertiesBegin;
  End = Begin ? PS->PropertiesEnd : nullptr;
}

void RTDeviceBinaryImage::print() const {
  std::cerr << "  --- Image " << Bin << "\n";
  if (!Bin)
    return;
  std::cerr << "    Version  : " << (int)Bin->Version << "\n";
  std::cerr << "    Kind     : " << (int)Bin->Kind << "\n";
  std::cerr << "    Format   : " << (int)Bin->Format << "\n";
  std::cerr << "    Target   : " << Bin->DeviceTargetSpec << "\n";
  std::cerr << "    Bin size : "
            << ((intptr_t)Bin->BinaryEnd - (intptr_t)Bin->BinaryStart) << "\n";
  std::cerr << "    Compile options : "
            << (Bin->CompileOptions ? Bin->CompileOptions : "NULL") << "\n";
  std::cerr << "    Link options    : "
            << (Bin->LinkOptions ? Bin->LinkOptions : "NULL") << "\n";
  std::cerr << "    Entries  : ";

  for (sycl_offload_entry EntriesIt = Bin->EntriesBegin;
       EntriesIt != Bin->EntriesEnd; EntriesIt = EntriesIt->Increment())
    std::cerr << EntriesIt->GetName() << " ";
  std::cerr << "\n";
  std::cerr << "    Properties [" << Bin->PropertySetsBegin << "-"
            << Bin->PropertySetsEnd << "]:\n";

  for (sycl_device_binary_property_set PS = Bin->PropertySetsBegin;
       PS != Bin->PropertySetsEnd; ++PS) {
    std::cerr << "      Category " << PS->Name << " [" << PS->PropertiesBegin
              << "-" << PS->PropertiesEnd << "]:\n";

    for (sycl_device_binary_property P = PS->PropertiesBegin;
         P != PS->PropertiesEnd; ++P) {
      std::cerr << "        " << DeviceBinaryProperty(P) << "\n";
    }
  }
}

void RTDeviceBinaryImage::dump(std::ostream &Out) const {
  size_t ImgSize = getSize();
  Out.write(reinterpret_cast<const char *>(Bin->BinaryStart), ImgSize);
}

sycl_device_binary_property
RTDeviceBinaryImage::getProperty(const char *PropName) const {
  RTDeviceBinaryImage::PropertyRange BoolProp;
  BoolProp.init(Bin, __SYCL_PROPERTY_SET_SYCL_MISC_PROP);
  if (!BoolProp.isAvailable())
    return nullptr;
  auto It = std::find_if(BoolProp.begin(), BoolProp.end(),
                         [=](sycl_device_binary_property Prop) {
                           return !strcmp(PropName, Prop->Name);
                         });
  if (It == BoolProp.end())
    return nullptr;

  return *It;
}

void RTDeviceBinaryImage::init(sycl_device_binary Bin) {
  // Bin != nullptr is guaranteed here.
  this->Bin = Bin;
  // If device binary image format wasn't set by its producer, then can't change
  // now, because 'Bin' data is part of the executable image loaded into memory
  // which can't be modified (easily).
  // TODO clang driver + ClangOffloadWrapper can figure out the format and set
  // it when invoking the offload wrapper job
  Format = static_cast<ur::DeviceBinaryType>(Bin->Format);

  // For compressed images, we delay determining the format until the image is
  // decompressed.
  if (Format == SYCL_DEVICE_BINARY_TYPE_NONE)
    // try to determine the format; may remain "NONE"
    Format = ur::getBinaryImageFormat(Bin->BinaryStart, getSize());

  SpecConstIDMap.init(Bin, __SYCL_PROPERTY_SET_SPEC_CONST_MAP);
  SpecConstDefaultValuesMap.init(
      Bin, __SYCL_PROPERTY_SET_SPEC_CONST_DEFAULT_VALUES_MAP);
  DeviceLibReqMask.init(Bin, __SYCL_PROPERTY_SET_DEVICELIB_REQ_MASK);
  DeviceLibMetadata.init(Bin, __SYCL_PROPERTY_SET_DEVICELIB_METADATA);
  KernelParamOptInfo.init(Bin, __SYCL_PROPERTY_SET_KERNEL_PARAM_OPT_INFO);
  AssertUsed.init(Bin, __SYCL_PROPERTY_SET_SYCL_ASSERT_USED);
  ImplicitLocalArg.init(Bin, __SYCL_PROPERTY_SET_SYCL_IMPLICIT_LOCAL_ARG);
  ProgramMetadata.init(Bin, __SYCL_PROPERTY_SET_PROGRAM_METADATA);
  // Convert ProgramMetadata into the UR format
  for (const auto &Prop : ProgramMetadata) {
    ProgramMetadataUR.push_back(
        ur::mapDeviceBinaryPropertyToProgramMetadata(Prop));
  }
  ExportedSymbols.init(Bin, __SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS);
  ImportedSymbols.init(Bin, __SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS);
  DeviceGlobals.init(Bin, __SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS);
  DeviceRequirements.init(Bin, __SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS);
  HostPipes.init(Bin, __SYCL_PROPERTY_SET_SYCL_HOST_PIPES);
  VirtualFunctions.init(Bin, __SYCL_PROPERTY_SET_SYCL_VIRTUAL_FUNCTIONS);
  RegisteredKernels.init(Bin, __SYCL_PROPERTY_SET_SYCL_REGISTERED_KERNELS);

  ImageId = ImageCounter++;
}

std::atomic<uintptr_t> RTDeviceBinaryImage::ImageCounter = 1;

DynRTDeviceBinaryImage::DynRTDeviceBinaryImage(
    std::unique_ptr<char[]> &&DataPtr, size_t DataSize)
    : RTDeviceBinaryImage() {
  Data = std::move(DataPtr);
  Bin = new sycl_device_binary_struct();
  Bin->Version = SYCL_DEVICE_BINARY_VERSION;
  Bin->Kind = SYCL_DEVICE_BINARY_OFFLOAD_KIND_SYCL;
  Bin->CompileOptions = "";
  Bin->LinkOptions = "";
  Bin->ManifestStart = nullptr;
  Bin->ManifestEnd = nullptr;
  Bin->BinaryStart = reinterpret_cast<unsigned char *>(Data.get());
  Bin->BinaryEnd = Bin->BinaryStart + DataSize;
  Bin->EntriesBegin = nullptr;
  Bin->EntriesEnd = nullptr;
  Bin->Format = ur::getBinaryImageFormat(Bin->BinaryStart, DataSize);
  switch (Bin->Format) {
  case SYCL_DEVICE_BINARY_TYPE_SPIRV:
    Bin->DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_SPIRV64;
    break;
  default:
    Bin->DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_UNKNOWN;
  }
  init(Bin);
}

DynRTDeviceBinaryImage::~DynRTDeviceBinaryImage() {
  delete Bin;
  Bin = nullptr;
}

#ifndef SYCL_RT_ZSTD_NOT_AVAIABLE
CompressedRTDeviceBinaryImage::CompressedRTDeviceBinaryImage(
    sycl_device_binary CompressedBin)
    : RTDeviceBinaryImage() {

  // 'CompressedBin' is part of the executable image loaded into memory
  // which can't be modified easily. So, we need to make a copy of it.
  Bin = new sycl_device_binary_struct(*CompressedBin);

  // Get the decompressed size of the binary image.
  m_ImageSize = ZSTDCompressor::GetDecompressedSize(
      reinterpret_cast<const char *>(Bin->BinaryStart),
      static_cast<size_t>(Bin->BinaryEnd - Bin->BinaryStart));

  init(Bin);
}

void CompressedRTDeviceBinaryImage::Decompress() {

  size_t CompressedDataSize =
      static_cast<size_t>(Bin->BinaryEnd - Bin->BinaryStart);

  size_t DecompressedSize = 0;
  m_DecompressedData = ZSTDCompressor::DecompressBlob(
      reinterpret_cast<const char *>(Bin->BinaryStart), CompressedDataSize,
      DecompressedSize);

  Bin->BinaryStart =
      reinterpret_cast<const unsigned char *>(m_DecompressedData.get());
  Bin->BinaryEnd = Bin->BinaryStart + DecompressedSize;

  Bin->Format = ur::getBinaryImageFormat(Bin->BinaryStart, getSize());
  Format = static_cast<ur::DeviceBinaryType>(Bin->Format);
}

CompressedRTDeviceBinaryImage::~CompressedRTDeviceBinaryImage() {
  // De-allocate device binary struct.
  delete Bin;
  Bin = nullptr;
}
#endif // SYCL_RT_ZSTD_NOT_AVAIABLE

} // namespace detail
} // namespace _V1
} // namespace sycl
