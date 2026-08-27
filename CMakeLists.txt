cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ProtoSYCL VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Clang REQUIRED)
find_package(LLVM REQUIRED CONFIG)
include(AddLLVM)

if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

# sycl++ Compiler Wrapper
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sycl++
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/sycl++
        ${CMAKE_CURRENT_BINARY_DIR}/sycl++
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/sycl++
    COMMENT "Copying sycl++ compiler wrapper to build directory"
)
add_custom_target(ProtoSYCLCompilerWrapper ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sycl++
)
set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_BINARY_DIR}/sycl++)

# Compiler Plugin
add_llvm_library(clangProtoSYCL SHARED src/compiler/clang_plugin.cpp)
target_include_directories(clangProtoSYCL PRIVATE ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_compile_definitions(clangProtoSYCL PRIVATE ${CLANG_DEFINITIONS} ${LLVM_DEFINITIONS})
if(APPLE)
    target_link_options(clangProtoSYCL PRIVATE -undefined dynamic_lookup)
endif()
add_dependencies(clangProtoSYCL ProtoSYCLCompilerWrapper)

# LLVM Pass Plugin
add_llvm_pass_plugin(llvmProtoSYCL src/compiler/llvm_plugin.cpp)
target_include_directories(llvmProtoSYCL PRIVATE ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_compile_definitions(llvmProtoSYCL PRIVATE ${CLANG_DEFINITIONS} ${LLVM_DEFINITIONS})
if(APPLE)
    target_link_options(llvmProtoSYCL PRIVATE -undefined dynamic_lookup)
endif()
add_dependencies(llvmProtoSYCL ProtoSYCLCompilerWrapper)

# Attribute Rewriter
add_library(ProtoSYCLRewriter SHARED
    src/compiler/attribute_rewriter.cpp
)
target_compile_options(ProtoSYCLRewriter PRIVATE -fPIC)
add_dependencies(ProtoSYCLRewriter ProtoSYCLCompilerWrapper)

# Compiler Dependencies
add_custom_target(ProtoSYCLCompilerDependencies ""
    DEPENDS
        clangProtoSYCL
        llvmProtoSYCL
        ProtoSYCLRewriter
        ProtoSYCLCompilerWrapper
)

# Runtime Library
add_library(ProtoSYCL STATIC
    include/CL/sycl.hpp
    include/sycl/sycl.hpp
    include/sycl/detail/access.hpp
    include/sycl/detail/accessor_tag.hpp
    include/sycl/detail/accessor.hpp
    include/sycl/detail/aspect.hpp
    include/sycl/detail/atomic_ref.hpp
    include/sycl/detail/atomic.hpp
    include/sycl/detail/backend.hpp
    include/sycl/detail/buffer.hpp
    include/sycl/detail/bundle_state.hpp
    include/sycl/detail/cl_types.hpp
    include/sycl/detail/context.hpp
    include/sycl/detail/device_event.hpp
    include/sycl/detail/device_image.hpp
    include/sycl/detail/device.hpp
    include/sycl/detail/errc.hpp
    include/sycl/detail/event.hpp
    include/sycl/detail/exception.hpp
    include/sycl/detail/function_objects.hpp
    include/sycl/detail/future.hpp
    include/sycl/detail/group_barrier.hpp
    include/sycl/detail/group.hpp
    include/sycl/detail/h_item.hpp
    include/sycl/detail/half.hpp
    include/sycl/detail/handler.hpp
    include/sycl/detail/host_accessor.hpp
    include/sycl/detail/id.hpp
    include/sycl/detail/image.hpp
    include/sycl/detail/is_device_copyable.hpp
    include/sycl/detail/item.hpp
    include/sycl/detail/kernel_bundle.hpp
    include/sycl/detail/kernel_id.hpp
    include/sycl/detail/kernel.hpp
    include/sycl/detail/local_accessor.hpp
    include/sycl/detail/marray.hpp
    include/sycl/detail/math_geometric.hpp
    include/sycl/detail/math_half.hpp
    include/sycl/detail/math_integer.hpp
    include/sycl/detail/math_native.hpp
    include/sycl/detail/math_relational.hpp
    include/sycl/detail/math.hpp
    include/sycl/detail/memory_order.hpp
    include/sycl/detail/memory_scope.hpp
    include/sycl/detail/multi_ptr.hpp
    include/sycl/detail/nd_item.hpp
    include/sycl/detail/nd_range.hpp
    include/sycl/detail/platform.hpp
    include/sycl/detail/private_memory.hpp
    include/sycl/detail/property_list.hpp
    include/sycl/detail/property.hpp
    include/sycl/detail/queue.hpp
    include/sycl/detail/range.hpp
    include/sycl/detail/reduction.hpp
    include/sycl/detail/specialization_id.hpp
    include/sycl/detail/stream.hpp
    include/sycl/detail/sub_group.hpp
    include/sycl/detail/usm.hpp
    include/sycl/detail/util.accessor_iterator.hpp
    include/sycl/detail/util.allocator.hpp
    include/sycl/detail/util.descriptor.hpp
    include/sycl/detail/util.forward.hpp
    include/sycl/detail/util.kernel_registry.hpp
    include/sycl/detail/util.linearize.hpp
    include/sycl/detail/util.subscript.hpp
    include/sycl/detail/util.task_graph.hpp
    include/sycl/detail/util.task.hpp
    include/sycl/detail/vec.hpp
    include/sycl/detail/khr/free_function_kernels.hpp
    src/runtime/context.cpp
    src/runtime/exception.cpp
    src/runtime/kernel_bundle.cpp
    src/runtime/platform.cpp
    src/runtime/stream.cpp
    src/runtime/usm.cpp
    src/runtime/util.kernel_registry.cpp
    src/runtime/util.task_graph.cpp
)
target_include_directories(ProtoSYCL PUBLIC include)
target_compile_options(ProtoSYCL
    INTERFACE
        -fplugin=$<TARGET_FILE:clangProtoSYCL>
        -Xclang -fpass-plugin=$<TARGET_FILE:llvmProtoSYCL>
)
add_dependencies(ProtoSYCL ProtoSYCLCompilerDependencies)

# Testing
enable_testing()

function(add_protosycl_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE ProtoSYCL)
    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction(add_protosycl_test)

add_protosycl_test(example_test test/example.cpp)
add_protosycl_test(typical_test test/typical.cpp)
add_protosycl_test(attrs_test test/attrs.cpp)
add_protosycl_test(free_function_kernels test/khr/free_function_kernels.cpp)
