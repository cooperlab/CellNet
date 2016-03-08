#
#	Copyright (c) 2015-2016, Emory University
#	All rights reserved.
#
#	Redistribution and use in source and binary forms, with or without modification, are
#	permitted provided that the following conditions are met:
#
#	1. Redistributions of source code must retain the above copyright notice, this list of
#	conditions and the following disclaimer.
#
#	2. Redistributions in binary form must reproduce the above copyright notice, this list
# 	of conditions and the following disclaimer in the documentation and/or other materials
#	provided with the distribution.
#
#	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
#	OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
#	SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#	BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#	WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#	DAMAGE.
#
#
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
set(TissueNet_VERSION_MAJOR 0)
set(TissueNet_VERSION_MINOR 9)



#
### CUDA
#
find_package( CUDA )

if( NOT CUDA_FOUND ) 
    message(WARNING "CUDA libraries not found - building CPU only")
else( NOT CUDA_FOUND )

	if( CUDA_VERSION_STRING VERSION_LESS "7.0" )
		message(FATAL_ERROR "TissueNet requires CUDA version 7.0 or greater")
	endif( CUDA_VERSION_STRING VERSION_LESS "7.0" )

	set(LIBRARIES ${LIBRARIES} ${CUDA_LIBRARIES})
	set(LIBRARIES ${LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
	include_directories(${CUDA_INCLUDE_DIRS})

endif( NOT CUDA_FOUND )



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
find_package( Openslide )

if( NOT Openslide_FOUND ) 
    message(WARNING "Openslide libraries not found - Not building convert tools")
else( NOT Openslide_FOUND )

	include_directories(${Openslide_INCLUDE_DIRS})
	set(LIBRARIES ${LIBRARIES} ${Openslide_LIBRARIES})
	set(CONVERT_SLIDE_LIBS ${CONVERT_SLIDE_LIBS} ${Openslide_LIBRARIES})

endif( NOT Openslide_FOUND )



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
## Generate gengetopt files and commandline .c & .h for each target
#
configure_file("${PROJECT_SOURCE_DIR}/src/train-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/train-opt.ggo")
configure_file("${PROJECT_SOURCE_DIR}/src/predict-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/predict-opt.ggo")
configure_file("${PROJECT_SOURCE_DIR}/src/img-dump-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/img-dump-opt.ggo")


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
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/img-dump-cmd.c ${CMAKE_CURRENT_BINARY_DIR}/img-dump-cmd.h
	COMMAND ${GENGETOPT}
	ARGS	-i ${CMAKE_CURRENT_BINARY_DIR}/img-dump-opt.ggo -F img-dump-cmd
	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/img-dump-opt.ggo
)


#
## Also for convert targets if openslide installed
#
if( Openslide_FOUND ) 
	configure_file("${PROJECT_SOURCE_DIR}/src/data-conv-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/data-conv-opt.ggo")
	configure_file("${PROJECT_SOURCE_DIR}/src/slide-conv-opt.ggo.in" "${CMAKE_CURRENT_BINARY_DIR}/slide-conv-opt.ggo")


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

endif( Openslide_FOUND )





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
				grayscale_node.cpp prediction_node.cpp augmentation_node.cpp laplacian_pyramid_node.cpp
				multires_node.cpp ${CMAKE_CURRENT_BINARY_DIR}/predict-cmd.c)


target_link_libraries(PREDICT ${LIBRARIES})
set_target_properties(PREDICT PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app)
set_target_properties(PREDICT PROPERTIES OUTPUT_NAME tn_predict${CMAKE_DEBUG_POSTFIX})


#
## tn_train target
#
add_executable(TRAIN train.cpp utils.cpp read_hdf5_node.cpp graph_net.cpp node.cpp edge.cpp 
			   grayscale_node.cpp train_node.cpp augmentation_node.cpp laplacian_pyramid_node.cpp
				multires_node.cpp ${CMAKE_CURRENT_BINARY_DIR}/train-cmd.c)

target_link_libraries(TRAIN ${LIBRARIES})
set_target_properties(TRAIN PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app)
set_target_properties(TRAIN PROPERTIES OUTPUT_NAME tn_train${CMAKE_DEBUG_POSTFIX})



#
#	Set train & predict to CPU only if CUDA libraries not found
#
if( NOT CUDA_FOUND )
	target_compile_options(TRAIN PUBLIC "-DCPU_ONLY")
	target_compile_options(PREDICT PUBLIC "-DCPU_ONLY")
endif( NOT CUDA_FOUND)


#
## img_dump target
#
add_executable(IMG_DUMP imgdump.cpp read_hdf5_node.cpp write_image_node.cpp node.cpp graph_net.cpp 
						edge.cpp grayscale_node.cpp augmentation_node.cpp utils.cpp 
						${CMAKE_CURRENT_BINARY_DIR}/img-dump-cmd.c)


target_link_libraries(IMG_DUMP ${LIBRARIES})
set_target_properties(IMG_DUMP PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app)
set_target_properties(IMG_DUMP PROPERTIES OUTPUT_NAME tn_image_dump${CMAKE_DEBUG_POSTFIX})


if( Openslide_FOUND ) 
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


	#
	## Install all targets
	#
	install(TARGETS TRAIN PREDICT CONVERT_SLIDE CONVERT_DATASET IMG_DUMP
			RUNTIME DESTINATION bin)

elseif( NOT Openslide_FOUND ) 


	#
	## Only install image dume, train and predict targets
	#
	install(TARGETS TRAIN PREDICT IMG_DUMP
			RUNTIME DESTINATION bin)
			
endif( Openslide_FOUND ) 
