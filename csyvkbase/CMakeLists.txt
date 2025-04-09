file(GLOB CSY_VK_BASE_SRC "*.cpp" "*.hpp" "*.h" "*.inl" "../external/imgui/*.cpp")
file(GLOB CSY_VK_BASE_HEADERS "*.hpp" "*.h" "*.inl")

set(KTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external/ktx)
set(KTX_SOURCES
    ${KTX_DIR}/lib/texture.c
    ${KTX_DIR}/lib/hashlist.c
    ${KTX_DIR}/lib/checkheader.c
    ${KTX_DIR}/lib/swap.c
    ${KTX_DIR}/lib/memstream.c
    ${KTX_DIR}/lib/filestream.c)

add_library(csyvkbase STATIC ${CSY_VK_BASE_SRC} ${KTX_SOURCES})
if(WIN32)
    target_link_libraries(csyvkbase ${Vulkan_LIBRARY} ${WINLIBS})
 else(WIN32)
    target_link_libraries(csyvkbase ${Vulkan_LIBRARY} ${XCB_LIBRARIES} ${WAYLAND_CLIENT_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif(WIN32)