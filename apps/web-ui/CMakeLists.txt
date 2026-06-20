set(VPIPE_WEBUI_SOURCES
    main.cc
    http-server.cc
    session-api.cc
    web-ui-delegate.cc
    web-ui-log-delegate.cc
    db-browser.cc
    system-status.cc
    startup-checks.cc
)
# Microphone permission probe (AVCaptureDevice authorization status) is
# Objective-C++; Apple-only. The non-Apple fallback lives in
# startup-checks.cc (#if !defined(__APPLE__)).
if(APPLE)
    list(APPEND VPIPE_WEBUI_SOURCES startup-checks-mac.mm)
endif()

# Embed the web/ asset tree into the binary so a packaged build serves
# the UI from memory (no doc-root on disk required). The filesystem
# doc-root still wins when present, so a dev can live-edit web/ files
# (see the precedence in main.cc). CONFIGURE_DEPENDS re-globs at build
# time, so adding/removing an asset file is picked up automatically; an
# edit to an existing file triggers regeneration via the DEPENDS list.
file(GLOB_RECURSE VPIPE_WEBUI_ASSET_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/web/*")
set(VPIPE_WEBUI_ASSETS_CC
    "${CMAKE_CURRENT_BINARY_DIR}/web-assets-generated.cc")
add_custom_command(
    OUTPUT  "${VPIPE_WEBUI_ASSETS_CC}"
    COMMAND "${CMAKE_COMMAND}"
            -D "WEB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/web"
            -D "OUTPUT=${VPIPE_WEBUI_ASSETS_CC}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed-web-assets.cmake"
    DEPENDS ${VPIPE_WEBUI_ASSET_FILES}
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed-web-assets.cmake"
    COMMENT "Embedding web-ui assets into the binary"
    VERBATIM)

add_executable(vpipe-web-ui
    ${VPIPE_WEBUI_SOURCES}
    ${VPIPE_WEBUI_ASSETS_CC})

target_include_directories(vpipe-web-ui
    PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)

# lmdb for the read-only Database browser (db-browser.cc). The lmdb
# target exposes its headers as a PUBLIC (SYSTEM) include dir; vpipe
# links it PRIVATE so it isn't propagated -- link it here too.
target_link_libraries(vpipe-web-ui PRIVATE vpipe lmdb)

# system-status.cc reads Apple Silicon GPU stats via IOKit
# (IOAccelerator's PerformanceStatistics) -- the same data nvtop's
# AS backend and asitop / mactop consume. CoreFoundation is already
# brought in transitively via libvpipe; IOKit is local to this app.
if(APPLE)
    # IOKit for GPU stats. The IOReport private framework (used for
    # ANE energy samples, the same source macmon reads) is resolved at
    # runtime via dlopen("/System/Library/PrivateFrameworks/IOReport
    # .framework/IOReport") from the dyld shared cache -- no link-time
    # framework path exists for it on modern macOS, so we don't pass
    # `-framework IOReport`.
    target_link_libraries(vpipe-web-ui PRIVATE
        "-framework IOKit"
        "-framework CoreFoundation")
    # AVFoundation for the microphone authorization-status probe;
    # CoreAudio for the audio-input device enumeration
    # (startup-checks-mac.mm).
    target_link_libraries(vpipe-web-ui PRIVATE
        "-framework AVFoundation"
        "-framework CoreAudio")
endif()

install(TARGETS vpipe-web-ui RUNTIME DESTINATION bin)
# The UI assets are embedded in the binary (above), so the installed
# executable is self-contained -- the raw asset tree is NOT installed by
# default. Turn this on to also ship the tree (e.g. as an override source
# for $VPIPE_WEBUI_DOCROOT to patch the UI without a rebuild).
option(VPIPE_WEBUI_INSTALL_ASSETS
       "Also install the web-ui asset tree to share/vpipe/web-ui" OFF)
if(VPIPE_WEBUI_INSTALL_ASSETS)
    install(DIRECTORY web/ DESTINATION share/vpipe/web-ui)
endif()
