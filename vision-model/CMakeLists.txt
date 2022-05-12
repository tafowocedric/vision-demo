# add torch lib
link_libraries(${TORCH_LIBRARIES})

file(GLOB SOURCES "*.hpp")

# add libraries
add_library(vision-model Detector.cpp ${SOURCES})

##################################################
################### Backbone #####################
# add subdirectory
add_subdirectory(backbone)

# vision model location
target_include_directories(vision-model PUBLIC backbone)
target_link_directories(vision-model PRIVATE backbone)

# target lib file
target_link_libraries(vision-model PRIVATE backbone)

##################################################
##################### END  #######################


##################################################
################# Architecture ###################
# add subdirectory
add_subdirectory(architecture)

# vision model location
target_include_directories(vision-model PUBLIC architecture)
target_link_directories(vision-model PRIVATE architecture)

# target lib file
target_link_libraries(vision-model PRIVATE architecture)

##################################################
##################### END  #######################