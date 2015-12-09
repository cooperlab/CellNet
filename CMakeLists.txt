#
#	Copyright (c) 2015, Emory University
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
cmake_minimum_required(VERSION 3.0)
project(TissueNet)

#
# Needed to add -Wl,--no-as-needed to allow std::threads to work. Not
# sure why, al_serverd from the active learning project works without the
# flags. 
#
if( CMAKE_COMPILER_IS_GNUCC )
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -Wl,--no-as-needed")
else( CMAKE_COMPILER_IS_GNUCC )
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
endif( CMAKE_COMPILER_IS_GNUCC )


if( CMAKE_BUILD_TYPE STREQUAL Debug ) 
	set(TissueNet_DEBUG 1)
    set(CMAKE_DEBUG_POSTFIX "_d")
else( CMAKE_BUILD_TYPE STREQUAL Debug)
	set(TissueNet_DEBUG 0)
    set(CMAKE_DEBUG_POSTFIX "")
endif( CMAKE_BUILD_TYPE STREQUAL Debug )


add_subdirectory(src)

