cmake_minimum_required(VERSION 3.10)

# set the project name
project(visionai_core)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

##################################################
#################### Torch #####################
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/dist/lib/libtorch)

set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist/lib/libtorch/share/cmake/Torch)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/dist/lib/libtorch)

find_package(Torch QUIET PATHS ${CMAKE_SOURCE_DIR}/libtorch)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++11")

set(TORCH_INCLUDE_DIRS ${TORCH_INSTALL_PREFIX}/include ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)

##################################################
################### OpenCV #######################
find_package(OpenCV REQUIRED)


message(STATUS ">>    Working Dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS ">>    Torch: ${Torch_DIR}")
message(STATUS ">>    OpenCV library status: ")
message(STATUS ">>    version: ${OpenCV_VERSION}")
message(STATUS ">>    opencv path: ${OpenCV_DIR}")
message(STATUS ">>    dir: ${OpenCV_LIB_DIR}")
message(STATUS ">>    libraries: ${OpenCV_LIBS}")
message(STATUS ">>    include path: ${OpenCV_INCLUDE_DIRS}")


# add the executable
add_executable(${PROJECT_NAME} main.cpp)


##################################################
#################### OPENCV ######################
include_directories("/usr/local/include/")
include_directories(${OpenCV_INCLUDE_DIRS})

# target opencv4 lib to executable
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

##################################################
##################### END  #######################


##################################################
#################### TORCH ######################
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

##################################################
##################### END  #######################


##################################################
######### INCLUDE CUSTOME HEARDER FILES ##########
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dist/include/)

##################################################
##################### END  #######################


##################################################
################# VISION MODEL ###################
# add subdirectory
add_subdirectory(vision-model)

# vision model location
target_include_directories(${PROJECT_NAME} PUBLIC vision-model)
target_link_directories(${PROJECT_NAME} PRIVATE vision-model)

# target lib file
target_link_libraries(${PROJECT_NAME} PRIVATE vision-model)

##################################################
##################### END  #######################


##################################################
############### VISION PIPELINE ##################
# add subdirectory
add_subdirectory(vision-pipeline)

# vision pipeline location
target_include_directories(${PROJECT_NAME} PUBLIC vision-pipeline)
target_link_directories(${PROJECT_NAME} PRIVATE vision-pipeline)

# target lib file
target_link_libraries(${PROJECT_NAME} PRIVATE vision-pipeline)

##################################################
##################### END  #######################