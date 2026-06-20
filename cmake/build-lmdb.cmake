set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(LMDB_SRC_DIR ${CMAKE_SOURCE_DIR}/extern/lmdb/libraries/liblmdb)

add_library(lmdb STATIC
    ${LMDB_SRC_DIR}/mdb.c
    ${LMDB_SRC_DIR}/midl.c
)
target_include_directories(lmdb SYSTEM PUBLIC ${LMDB_SRC_DIR})
target_link_libraries(lmdb PUBLIC Threads::Threads)
set_target_properties(lmdb PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99
)
