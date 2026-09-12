cmake_minimum_required(VERSION 3.22)
project(FssrAmpDemo VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(NAM_CORE_ROOT "${REPO_ROOT}/third_party/NeuralAmpModelerCore")
file(GLOB NAM_SOURCES "${NAM_CORE_ROOT}/NAM/*.cpp" "${NAM_CORE_ROOT}/NAM/*.c")

add_subdirectory("${REPO_ROOT}/third_party/JUCE" juce EXCLUDE_FROM_ALL)

juce_add_plugin(FssrAmpDemo
  PRODUCT_NAME "FSSR Amp Demo"
  COMPANY_NAME "FSSR-NAM"
  PLUGIN_MANUFACTURER_CODE Fssr
  PLUGIN_CODE Fsad
  FORMATS VST3 Standalone
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  COPY_PLUGIN_AFTER_BUILD FALSE
)

target_sources(FssrAmpDemo PRIVATE
  src/PluginProcessor.cpp
  src/PluginEditor.cpp
  ${NAM_SOURCES}
)

target_include_directories(FssrAmpDemo SYSTEM PRIVATE
  "${NAM_CORE_ROOT}"
  "${NAM_CORE_ROOT}/Dependencies/eigen"
  "${NAM_CORE_ROOT}/Dependencies/nlohmann"
)

target_compile_definitions(FssrAmpDemo PUBLIC
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  JUCE_VST3_CAN_REPLACE_VST2=0
  NAM_ENABLE_A2_FAST
)

# -O3 without fast-math: the demo claims NaN propagation and IEEE parity.
target_compile_options(FssrAmpDemo PRIVATE "$<$<CONFIG:Release>:-O3>")

target_link_libraries(FssrAmpDemo PRIVATE
  juce::juce_audio_utils
  juce::juce_dsp
  juce::juce_recommended_config_flags
  juce::juce_recommended_warning_flags
)
