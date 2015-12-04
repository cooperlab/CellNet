set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
set(TissueNet_VERSION_MAJOR 0)
set(TissueNet_VERSION_MINOR 2)




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
set(CONVERT_SLIDE_LIBS ${CONVERT_SLIDE_LIBS} ${HDF5_LIBRARIES})
set(CONVERT_SLIDE_LIBS ${CONVERT_SLIDE_LIBS} ${HDF5_HL_LIBRARIES})



#
### OpenCV
#
find_package( OpenCV REQUIRED )

if( NOT OpenCV_FOUND ) 
    message(FATAL_ERROR "OpenCV libraries not found")
endif( NOT OpenCV_FOUND )

include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${OpenCV_LIBS})
set(CONVERT_SLIDE_LIBS ${CONVERT_SLIDE_LIBS} ${OpenCV_LIBS})

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
## Openslide
#
find_package( Openslide REQUIRED )

if( NOT Openslide_FOUND ) 
    message(FATAL_ERROR "Openslide libraries not found")
endif( NOT Openslide_FOUND )

include_directories(${Openslide_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${Openslide_LIBRARIES})
set(CONVERT_SLIDE_LIBS ${CONVERT_SLIDE_LIBS} ${Openslide_LIBRARIES})


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
## gengetopt - for generating command line options
#
find_program(GENGETOPT gengetopt)
if( GENGETOPT STREQUAL "GENGETOPT-NOTFOUND" )
	message(FATAL_ERROR "Unable to find gengetopt")
endif( GENGETOPT STREQUAL "GENGETOPT-NOTFOUND" )



#
## Generate gengetopt files for each target
#
configure_file("${PROJECT_SOURCE_DIR}/src/train-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/train-opt.ggo")
configure_file("${PROJECT_SOURCE_DIR}/src/predict-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/predict-opt.ggo")
configure_file("${PROJECT_SOURCE_DIR}/src/data-conv-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/data-conv-opt.ggo")
configure_file("${PROJECT_SOURCE_DIR}/src/slide-conv-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/slide-conv-opt.ggo")



#
##	Generate commandline .c & .h files for each target
#
ADD_CUSTOM_COMMAND(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/train-cmd.c ${CMAKE_CURRENT_BINARY_DIR}/train-cmd.h
	COMMAND ${GENGETOPT}
	ARGS	-i ${CMAKE_CURRENT_BINARY_DIR}/train-opt.ggo -F train-cmd
	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/train-opt.ggo
)


ADD_CUSTOM_COMMAND(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/predict-cmd.c ${CMAKE_CURRENT_BINARY_DIR}/predict-cmd.h
	COMMAND ${GENGETOPT}
	ARGS	-i ${CMAKE_CURRENT_BINARY_DIR}/predict-opt.ggo -F predict-cmd
	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/predict-opt.ggo
)


ADD_CUSTOM_COMMAND(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/data-conv-cmd.c ${CMAKE_CURRENT_BINARY_DIR}/data-conv-cmd.h
	COMMAND ${GENGETOPT}
	ARGS	-i ${CMAKE_CURRENT_BINARY_DIR}/data-conv-opt.ggo -F data-conv-cmd
	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/data-conv-opt.ggo
)


ADD_CUSTOM_COMMAND(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/slide-conv-cmd.c ${CMAKE_CURRENT_BINARY_DIR}/slide-conv-cmd.h
	COMMAND ${GENGETOPT}
	ARGS	-i ${CMAKE_CURRENT_BINARY_DIR}/slide-conv-opt.ggo -F slide-conv-cmd
	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/slide-conv-opt.ggo
)



#
## Generate base_config.h and add to include path
# 
configure_file("${PROJECT_SOURCE_DIR}/src/base_config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/base_config.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
## Include files are is a seperate directory
#
include_directories(${CMAKE_SOURCE_DIR}/include)


#
## tn_predict target
#
add_executable(PREDICT predict.cpp utils.cpp read_hdf5_node.cpp graph_net.cpp node.cpp edge.cpp
				grayscale_node.cpp prediction_node.cpp augmentation_node.cpp 
				${CMAKE_CURRENT_BINARY_DIR}/predict-cmd.c)


target_link_libraries(PREDICT ${LIBRARIES})
set_target_properties(PREDICT PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app)
set_target_properties(PREDICT PROPERTIES OUTPUT_NAME tn_predict${CMAKE_DEBUG_POSTFIX})


#
## tn_train target
#
add_executable(TRAIN train.cpp utils.cpp read_hdf5_node.cpp graph_net.cpp node.cpp edge.cpp 
			   grayscale_node.cpp train_node.cpp augmentation_node.cpp
				${CMAKE_CURRENT_BINARY_DIR}/train-cmd.c)

target_link_libraries(TRAIN ${LIBRARIES})
set_target_properties(TRAIN PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app)
set_target_properties(TRAIN PROPERTIES OUTPUT_NAME tn_train${CMAKE_DEBUG_POSTFIX})


#
## tn_convert_slide target
#
add_executable(CONVERT_SLIDE slide_convert.cpp utils.cpp ${CMAKE_CURRENT_BINARY_DIR}/slide-conv-cmd.c)

target_link_libraries(CONVERT_SLIDE ${CONVERT_SLIDE_LIBS})
set_target_properties(CONVERT_SLIDE PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app)
set_target_properties(CONVERT_SLIDE PROPERTIES OUTPUT_NAME tn_convert_slide${CMAKE_DEBUG_POSTFIX})


#
## tn_dataset_convert target
#
add_executable(CONVERT_DATASET dataset_convert.cpp utils.cpp ${CMAKE_CURRENT_BINARY_DIR}/data-conv-cmd.c)

target_link_libraries(CONVERT_DATASET ${CONVERT_SLIDE_LIBS})
set_target_properties(CONVERT_DATASET PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app)
set_target_properties(CONVERT_DATASET PROPERTIES OUTPUT_NAME tn_convert_dataset${CMAKE_DEBUG_POSTFIX})


