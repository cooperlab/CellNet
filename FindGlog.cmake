# - Find Glog {Google logger}  (includes and libraries)
#
# This module defines
#  Glog_INCLUDE_DIRS
#  Glog_LIBRARIES
#  Glog_FOUND
#

set(GLOG_FIND_REQUIRED true)


find_path(Glog_INCLUDE_DIRS glog/logging.h
    /usr/include/
    /usr/local/include/
)


set(GLOG_NAMES ${GLOG_NAMES} glog )
find_library(GLOG_LIBRARY
    NAMES ${GLOG_NAMES}
    PATHS
    /usr/lib64/
    /usr/lib/
    /usr/local/lib64/
    /usr/local/lib/
)



if (GLOG_LIBRARY AND Glog_INCLUDE_DIRS )
    set(Glog_LIBRARIES ${GLOG_LIBRARY})
    set(Glog_FOUND true)
endif (GLOG_LIBRARY AND Glog_INCLUDE_DIRS )



# Hide in the cmake cache
mark_as_advanced(Glog_LIBRARIES)
