# - Find Caffe (includes and libraries)
#
# This module defines
#  Caffe_INCLUDE_DIRS
#  Caffe_LIBRARIES
#  Caffe_FOUND
#

set(CAFFE_FIND_REQUIRED true)


find_path(Caffe_INCLUDE_DIRS caffe/caffe.hpp
    /usr/include/
    /usr/local/include/
)


set(CAFFE_NAMES ${CAFFE_NAMES} caffe )
find_library(Caffe_LIBRARIES
    NAMES ${CAFFE_NAMES}
    PATHS
    /usr/lib64/
    /usr/lib/
    /usr/local/lib64/
    /usr/local/lib/
)



if (Caffe_LIBRARIES AND Caffe_INCLUDE_DIRS )
    set(Caffe_FOUND true)
endif (Caffe_LIBRARIES AND Caffe_INCLUDE_DIRS )


