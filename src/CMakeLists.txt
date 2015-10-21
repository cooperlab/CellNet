set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
set(CellNet_VERSION_MAJOR 0)
set(CellNet_VERSION_MINOR 1)




#
### CUDA
#
find_package( CUDA REQUIRED )

if( NOT CUDA_FOUND ) 
    message(FATAL_ERROR "CUDA libraries not found")
endif( NOT CUDA_FOUND )

if( CUDA_VERSION_STRING VERSION_LESS "7.0" )
	message(FATAL_ERROR "CellNet requires CUDA version 7.0 or greater")
endif( CUDA_VERSION_STRING VERSION_LESS "7.0" )

include_directories(${CUDA_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${CUDA_LIBRARIES})
set(LIBRARIES ${LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})




#
### HDF5 package
#
find_package(HDF5 REQUIRED COMPONENTS HL)


if( NOT HDF5_FOUND )
	message(FATAL_ERROR "HDF5 libraries not found")
endif( NOT HDF5_FOUND )

include_directories(${HDF5_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${HDF5_LIBRARIES})
set(LIBRARIES ${LIBRARIES} ${HDF5_HL_LIBRARIES})




#
### OpenCV
#
find_package( OpenCV REQUIRED )

if( NOT OpenCV_FOUND ) 
    message(FATAL_ERROR "OpenCV libraries not found")
endif( NOT OpenCV_FOUND )

include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${OpenCV_LIBS})

add_definitions(-DUSE_OPENCV)




#
### Boost
#
find_package( Boost REQUIRED COMPONENTS thread system )

if( NOT Boost_FOUND ) 
    message(FATAL_ERROR "Boost libraries not found")
endif( NOT Boost_FOUND )

if( Boost_VERSION VERSION_LESS "1.55" )
	message(FATAL_ERROR "Caffe requires Boost version 1.55 or greater")
endif( Boost_VERSION VERSION_LESS "1.55" )

include_directories(${Boost_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${Boost_LIBRARIES})




#
## Glog
#
find_package( Glog REQUIRED )

if( NOT Glog_FOUND ) 
    message(FATAL_ERROR "Glog libraries not found")
endif( NOT Glog_FOUND )

include_directories(${Glog_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${Glog_LIBRARIES})




#
## Caffe 
#
find_package( Caffe REQUIRED )

if( NOT Caffe_FOUND ) 
    message(FATAL_ERROR "Caffe libraries not found")
endif( NOT Caffe_FOUND )

include_directories(${Caffe_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${Caffe_LIBRARIES})




#
## Generate base_config.h
# 
configure_file("${PROJECT_SOURCE_DIR}/src/base_config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/base_config.h")


include_directories(${CMAKE_SOURCE_DIR}/include)
#
## predict target
#
add_executable(PREDICT predict.cpp utils.cpp read_jpg_node.cpp graph_net.cpp node.cpp edge.cpp
				grayscale_node.cpp prediction_node.cpp)


target_link_libraries(PREDICT ${LIBRARIES})
set_target_properties(PREDICT PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app)
set_target_properties(PREDICT PROPERTIES OUTPUT_NAME predict${CMAKE_DEBUG_POSTFIX})


