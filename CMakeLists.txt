cmake_minimum_required(VERSION 3.0)
project(TissueNet)

#
# Needed to add -Wl,--no-as-needed to allow std::threads to work. Not
# sure why, al_serverd from the active learning project works without the
# flags. 
#
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11  -Wl,--no-as-needed")

if( CMAKE_BUILD_TYPE STREQUAL Debug ) 
	set(TissueNet_DEBUG 1)
    set(CMAKE_DEBUG_POSTFIX "_d")
else( CMAKE_BUILD_TYPE STREQUAL Debug)
	set(TissueNet_DEBUG 0)
    set(CMAKE_DEBUG_POSTFIX "")
endif( CMAKE_BUILD_TYPE STREQUAL Debug )


add_subdirectory(src)

