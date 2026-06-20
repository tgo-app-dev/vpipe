add_executable(vpipe-db-log-reader main.cc)

target_include_directories(vpipe-db-log-reader
    PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(vpipe-db-log-reader PRIVATE vpipe lmdb)

install(TARGETS vpipe-db-log-reader RUNTIME DESTINATION bin)
