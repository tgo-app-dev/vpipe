set(PUGIXML_SRC_DIR ${CMAKE_SOURCE_DIR}/extern/pugixml/src)

add_library(pugixml STATIC
    ${PUGIXML_SRC_DIR}/pugixml.cpp
)
target_include_directories(pugixml SYSTEM PUBLIC ${PUGIXML_SRC_DIR})
set_target_properties(pugixml PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
)
