# Include the headers from the "include" directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# CMakeList.txt : src file configuration
#

# List the source files for your framework components
set(SRC_FILES
    SimpleNeuralNet.cpp
    Layers/Dense.cpp
    # Add more source files as needed
)

# Create a library target for your framework
#add_library(SimpleNeuralNet ${SRC_FILES})

add_executable("SimpleNeuralNet" "SimpleNeuralNet.cpp")


# Specify the link dependencies, if any
# target_link_libraries(SimpleNeuralNetLib some_other_library)

# If you need to install the library
# install(TARGETS SimpleNeuralNetLib DESTINATION lib)
# install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)