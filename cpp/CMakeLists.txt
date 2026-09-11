cmake_minimum_required(VERSION 3.20)
project(fssr_nam_tools LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NAM_CORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/NeuralAmpModelerCore")
file(GLOB NAM_SOURCES_TOP "${NAM_CORE_ROOT}/NAM/*.cpp" "${NAM_CORE_ROOT}/NAM/*.c")
file(GLOB NAM_SOURCES_SUB "${NAM_CORE_ROOT}/NAM/*/*.cpp")
set(NAM_SOURCES ${NAM_SOURCES_TOP} ${NAM_SOURCES_SUB})

# -Ofast implies -ffast-math, which breaks NaN propagation and IEEE parity.
# The product lane builds with -O3 so its parity and NaN guarantees hold.
option(NAM_TOOLS_FAST_MATH "Build the NAM tools with -Ofast" ON)

function(add_nam_tool target source)
  add_executable(${target} ${source} ${NAM_SOURCES})
  target_compile_definitions(${target} PRIVATE NAM_ENABLE_A2_FAST)
  target_include_directories(${target} SYSTEM PRIVATE
    "${NAM_CORE_ROOT}"
    "${NAM_CORE_ROOT}/Dependencies/eigen"
    "${NAM_CORE_ROOT}/Dependencies/nlohmann"
  )
  if(NAM_TOOLS_FAST_MATH)
    target_compile_options(${target} PRIVATE "$<$<CONFIG:Release>:-Ofast>")
  else()
    target_compile_options(${target} PRIVATE "$<$<CONFIG:Release>:-O3>")
  endif()
  set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${target} PRIVATE stdc++fs)
  endif()
endfunction()

add_nam_tool(nam_block_runner tests/nam_block_runner.cpp)
add_nam_tool(nam_benchmark benchmarks/nam_benchmark.cpp)

add_library(fssr_r1_native STATIC src/fssr_r1_native.cpp)
target_include_directories(fssr_r1_native
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${NAM_CORE_ROOT}/Dependencies/nlohmann"
)
set_target_properties(fssr_r1_native PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
target_compile_options(fssr_r1_native PRIVATE "$<$<CONFIG:Release>:-Ofast>")
set_target_properties(fssr_r1_native PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

add_executable(r1_block_runner tests/r1_block_runner.cpp)
target_link_libraries(r1_block_runner PRIVATE fssr_r1_native)
set_target_properties(r1_block_runner PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

add_executable(r2_block_runner tests/r1_block_runner.cpp)
target_link_libraries(r2_block_runner PRIVATE fssr_r1_native)
set_target_properties(r2_block_runner PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

add_executable(r1_benchmark benchmarks/r1_benchmark.cpp)
target_link_libraries(r1_benchmark PRIVATE fssr_r1_native)
set_target_properties(r1_benchmark PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

add_executable(r2_benchmark benchmarks/r2_interleaved_benchmark.cpp ${NAM_SOURCES})
target_compile_definitions(r2_benchmark PRIVATE NAM_ENABLE_A2_FAST)
target_include_directories(r2_benchmark SYSTEM PRIVATE
  "${NAM_CORE_ROOT}"
  "${NAM_CORE_ROOT}/Dependencies/eigen"
  "${NAM_CORE_ROOT}/Dependencies/nlohmann"
)
target_link_libraries(r2_benchmark PRIVATE fssr_r1_native)
target_compile_options(r2_benchmark PRIVATE "$<$<CONFIG:Release>:-Ofast>")
set_target_properties(r2_benchmark PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  INTERPROCEDURAL_OPTIMIZATION TRUE
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(r2_benchmark PRIVATE stdc++fs)
endif()

add_library(amp_arch_skeleton STATIC src/amp_arch_skeleton.cpp)
target_include_directories(amp_arch_skeleton
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(amp_arch_skeleton PRIVATE
  "$<$<CONFIG:Release>:-Ofast;-march=native>"
)
set_target_properties(amp_arch_skeleton PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  INTERPROCEDURAL_OPTIMIZATION TRUE
)

add_executable(amp_arch_skeleton_runner tests/amp_arch_skeleton_runner.cpp)
target_link_libraries(amp_arch_skeleton_runner PRIVATE amp_arch_skeleton)
target_compile_options(amp_arch_skeleton_runner PRIVATE
  "$<$<CONFIG:Release>:-Ofast;-march=native>"
)
set_target_properties(amp_arch_skeleton_runner PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  INTERPROCEDURAL_OPTIMIZATION TRUE
)

add_executable(
  amp_arch_skeleton_benchmark
  benchmarks/amp_arch_skeleton_benchmark.cpp
  ${NAM_SOURCES}
)
target_compile_definitions(amp_arch_skeleton_benchmark PRIVATE NAM_ENABLE_A2_FAST)
target_include_directories(amp_arch_skeleton_benchmark SYSTEM PRIVATE
  "${NAM_CORE_ROOT}"
  "${NAM_CORE_ROOT}/Dependencies/eigen"
  "${NAM_CORE_ROOT}/Dependencies/nlohmann"
)
target_link_libraries(amp_arch_skeleton_benchmark PRIVATE amp_arch_skeleton)
target_compile_options(amp_arch_skeleton_benchmark PRIVATE
  "$<$<CONFIG:Release>:-Ofast;-march=native>"
)
set_target_properties(amp_arch_skeleton_benchmark PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  INTERPROCEDURAL_OPTIMIZATION TRUE
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(amp_arch_skeleton_benchmark PRIVATE stdc++fs)
endif()

add_library(sota_prototype STATIC src/sota_prototype.cpp)
target_include_directories(sota_prototype
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(sota_prototype PRIVATE
  "$<$<CONFIG:Release>:-Ofast;-march=native>"
)
set_target_properties(sota_prototype PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  INTERPROCEDURAL_OPTIMIZATION TRUE
)

add_executable(sota_block_runner tests/sota_block_runner.cpp)
target_link_libraries(sota_block_runner PRIVATE sota_prototype)
target_compile_options(sota_block_runner PRIVATE
  "$<$<CONFIG:Release>:-Ofast;-march=native>"
)
set_target_properties(sota_block_runner PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  INTERPROCEDURAL_OPTIMIZATION TRUE
)

add_executable(
  sota_interleaved_benchmark
  benchmarks/sota_interleaved_benchmark.cpp
)
target_link_libraries(sota_interleaved_benchmark PRIVATE sota_prototype)
target_compile_options(sota_interleaved_benchmark PRIVATE
  "$<$<CONFIG:Release>:-Ofast;-march=native>"
)
set_target_properties(sota_interleaved_benchmark PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  INTERPROCEDURAL_OPTIMIZATION TRUE
)

if(BUILD_TESTING)
  add_test(NAME sota_block_runner COMMAND sota_block_runner)
endif()
