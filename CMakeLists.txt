cmake_minimum_required(VERSION 3.0)
project(CellNet)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")

if( CMAKE_BUILD_TYPE STREQUAL Debug ) 
	set(CellNet_DEBUG 1)
    set(CMAKE_DEBUG_POSTFIX "_d")
else( CMAKE_BUILD_TYPE STREQUAL Debug)
	set(CellNet_DEBUG 0)
    set(CMAKE_DEBUG_POSTFIX "")
endif( CMAKE_BUILD_TYPE STREQUAL Debug )


add_subdirectory(src)

