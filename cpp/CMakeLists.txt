cmake_minimum_required(VERSION 3.20)
project(fssr_nam_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NAM_CORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/NeuralAmpModelerCore")
file(GLOB NAM_SOURCES_TOP "${NAM_CORE_ROOT}/NAM/*.cpp" "${NAM_CORE_ROOT}/NAM/*.c")
file(GLOB NAM_SOURCES_SUB "${NAM_CORE_ROOT}/NAM/*/*.cpp")
set(NAM_SOURCES ${NAM_SOURCES_TOP} ${NAM_SOURCES_SUB})

function(add_nam_tool target source)
  add_executable(${target} ${source} ${NAM_SOURCES})
  target_compile_definitions(${target} PRIVATE NAM_ENABLE_A2_FAST)
  target_include_directories(${target} SYSTEM PRIVATE
    "${NAM_CORE_ROOT}"
    "${NAM_CORE_ROOT}/Dependencies/eigen"
    "${NAM_CORE_ROOT}/Dependencies/nlohmann"
  )
  target_compile_options(${target} PRIVATE
    "$<$<CONFIG:Release>:-Ofast>"
  )
  set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${target} PRIVATE stdc++fs)
  endif()
endfunction()

add_nam_tool(nam_block_runner tests/nam_block_runner.cpp)
add_nam_tool(nam_benchmark benchmarks/nam_benchmark.cpp)
