# Invoked at build time via cmake -P with:
#   -DOUTPUT_FILE=<path to vpipe-version.h in build dir>
#   -DSOURCE_DIR=<project source root>
#   -DVERSION_MAJOR=<major>  -DVERSION_MINOR=<minor>

execute_process(
    COMMAND bash -c "git log -1 --pretty=%h | head -c 8"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_HASH_SHORT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND bash -c "git status -s | wc -l"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_DIRTY_COUNT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(STRIP "${GIT_DIRTY_COUNT}" GIT_DIRTY_COUNT)

if(NOT GIT_HASH_SHORT)
    set(GIT_HASH_SHORT "unknown0")
endif()
if(NOT GIT_DIRTY_COUNT)
    set(GIT_DIRTY_COUNT "0")
endif()

set(HEADER_CONTENT
"#pragma once
#define VPIPE_VERSION_MAJOR \"${VERSION_MAJOR}\"
#define VPIPE_VERSION_MINOR \"${VERSION_MINOR}\"
#define GIT_HASH \"${GIT_HASH_SHORT}*${GIT_DIRTY_COUNT}\"
")

# Only overwrite when content changes to avoid spurious recompilation of
# any TU that includes this header.
if(EXISTS "${OUTPUT_FILE}")
    file(READ "${OUTPUT_FILE}" _existing)
    if(_existing STREQUAL HEADER_CONTENT)
        return()
    endif()
endif()

file(WRITE "${OUTPUT_FILE}" "${HEADER_CONTENT}")
message(STATUS "vpipe-version.h: ${GIT_HASH_SHORT}*${GIT_DIRTY_COUNT}")
