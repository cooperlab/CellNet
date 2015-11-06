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
find_library(Glog_LIBRARIES
    NAMES ${GLOG_NAMES}
    PATHS
    /usr/lib64/
    /usr/lib/
    /usr/local/lib64/
    /usr/local/lib/
)



if (Glog_LIBRARIES AND Glog_INCLUDE_DIRS )
    set(Glog_FOUND true)
endif (Glog_LIBRARIES AND Glog_INCLUDE_DIRS )


