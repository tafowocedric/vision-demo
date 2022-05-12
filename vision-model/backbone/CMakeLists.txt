# add libraries
# file(GLOB SOURCES "*")
# list(FILTER SOURCES EXCLUDE REGEX ".*.txt$")

link_libraries(${TORCH_LIBRARIES})

# add torch lib
add_library(backbone VGG.cpp)
