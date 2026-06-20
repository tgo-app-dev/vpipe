# vpipe Python extension. Two artifacts land in
#   ${CMAKE_BINARY_DIR}/python/vpipe/
# so that `PYTHONPATH=${CMAKE_BINARY_DIR}/python python -c "import vpipe"`
# works straight out of the build tree:
#
#   _vpipe.<soabi>.so  -- the nanobind-compiled extension
#   __init__.py        -- thin wrapper that loads _vpipe and creates a
#                         default Session through SessionManager
set(VPIPE_PY_PKG_DIR ${CMAKE_BINARY_DIR}/python/vpipe)

nanobind_add_module(_vpipe
    NB_STATIC
    _vpipe.cc
)

target_include_directories(_vpipe
    PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(_vpipe PRIVATE vpipe)

set_target_properties(_vpipe PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${VPIPE_PY_PKG_DIR}
)

# Build-tree rpath so `import vpipe` in the build tree can find
# libvpipe without the user setting [DY]LD_LIBRARY_PATH.
if(APPLE)
    set_target_properties(_vpipe PROPERTIES
        BUILD_RPATH "@loader_path;@loader_path/../.."
        INSTALL_RPATH "@loader_path/../../.."
    )
elseif(UNIX)
    set_target_properties(_vpipe PROPERTIES
        BUILD_RPATH "$ORIGIN;$ORIGIN/../.."
        INSTALL_RPATH "$ORIGIN/../../.."
    )
endif()

# Drop the Python-side wrapper next to the compiled module.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/vpipe/__init__.py
    ${VPIPE_PY_PKG_DIR}/__init__.py
    COPYONLY
)

# Auto-generated stage reference. Globbed with CONFIGURE_DEPENDS so a
# new *-stage.h file triggers reconfigure on the next build. The
# Python script writes only when the rendered text changes, so an
# unrelated header touch doesn't invalidate the install timestamp.
file(GLOB_RECURSE VPIPE_STAGE_HEADERS
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/pipeline/*-stage.h
    ${CMAKE_SOURCE_DIR}/stages/*-stage.h
)

set(VPIPE_STAGES_MD ${VPIPE_PY_PKG_DIR}/STAGES.md)
add_custom_command(
    OUTPUT ${VPIPE_STAGES_MD}
    COMMAND ${Python_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/cmake/gen-stages-doc.py
        --out ${VPIPE_STAGES_MD}
        --source-root ${CMAKE_SOURCE_DIR}
        ${VPIPE_STAGE_HEADERS}
    DEPENDS ${VPIPE_STAGE_HEADERS}
        ${CMAKE_SOURCE_DIR}/cmake/gen-stages-doc.py
    COMMENT "Generating STAGES.md"
    VERBATIM
)
add_custom_target(vpipe_stages_doc ALL DEPENDS ${VPIPE_STAGES_MD})

# Auto-generated public-API reference. Order in VPIPE_PUBLIC_HEADERS
# is preserved in the rendered output, so list them in the order
# they should appear (umbrella header first, foundational types
# next, handle types, then session interface, then manager).
set(VPIPE_PUBLIC_HEADERS
    ${CMAKE_SOURCE_DIR}/include/vpipe/vpipe.h
    ${CMAKE_SOURCE_DIR}/include/vpipe/status.h
    ${CMAKE_SOURCE_DIR}/include/vpipe/pipeline-handle.h
    ${CMAKE_SOURCE_DIR}/include/vpipe/session-intf.h
    ${CMAKE_SOURCE_DIR}/include/vpipe/session-manager.h
)
set(VPIPE_API_MD ${VPIPE_PY_PKG_DIR}/API.md)
add_custom_command(
    OUTPUT ${VPIPE_API_MD}
    COMMAND ${Python_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/cmake/gen-api-doc.py
        --out ${VPIPE_API_MD}
        --source-root ${CMAKE_SOURCE_DIR}
        ${VPIPE_PUBLIC_HEADERS}
    DEPENDS ${VPIPE_PUBLIC_HEADERS}
        ${CMAKE_SOURCE_DIR}/cmake/gen-api-doc.py
    COMMENT "Generating API.md"
    VERBATIM
)
add_custom_target(vpipe_api_doc ALL DEPENDS ${VPIPE_API_MD})

# Install layout: lib/python<ver>/site-packages/vpipe/
set(VPIPE_PY_INSTALL_DIR
    lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/vpipe)
install(TARGETS _vpipe LIBRARY DESTINATION ${VPIPE_PY_INSTALL_DIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/vpipe/__init__.py
    DESTINATION ${VPIPE_PY_INSTALL_DIR})
install(FILES ${VPIPE_STAGES_MD} ${VPIPE_API_MD}
    DESTINATION ${VPIPE_PY_INSTALL_DIR})
