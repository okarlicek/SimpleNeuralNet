# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("SimpleNeuralNet")

# CMakeList.txt : CMake project for SimpleNeuralNet, include source and define
# project specific logic here.
#

# libs
add_subdirectory ("cmake")
GetEigen()

# src
add_subdirectory ("src")

# Add source to this project's executable.
# add_executable (SimpleNeuralNet "SimpleneuralNet.cpp" "SimpleNeuralNet.h" "src/Layers/BaseLayer.h"  "src/Layers/Dense.h" "src/Layers/Dense.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET SimpleNeuralNet PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.

