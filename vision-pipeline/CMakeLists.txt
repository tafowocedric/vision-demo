# add torch lib
link_libraries(${TORCH_LIBRARIES})

# add libraries
file(GLOB SOURCES "*.hpp")
# list(FILTER SOURCES EXCLUDE REGEX ".*utils.hpp$")

add_library(vision-pipeline ImagePreprocessing.cpp ImageAugmentation.cpp ${SOURCES})