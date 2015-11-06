# - Find Openslide (includes and libraries)
#
# This module defines
#  Openslide_INCLUDE_DIRS
#  Openslide_LIBRARIES
#  Openslide_FOUND
#

set(OPENSLIDE_FIND_REQUIRED true)


find_path(Openslide_INCLUDE_DIRS openslide/openslide.h
    /usr/include/
    /usr/local/include/
)


set(OPENSLIDE_NAMES ${OPENSLIDE_NAMES} openslide )
find_library(Openslide_LIBRARIES
    NAMES ${OPENSLIDE_NAMES}
    PATHS
    /usr/lib64/
    /usr/lib/
    /usr/local/lib64/
    /usr/local/lib/
)



if (Openslide_LIBRARIES AND Openslide_INCLUDE_DIRS )
    set(Openslide_FOUND true)
endif (Openslide_LIBRARIES AND Openslide_INCLUDE_DIRS )


