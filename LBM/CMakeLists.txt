cmake_minimum_required(VERSION 3.10)
project(LBM_Free_Surface)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义项目目录
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

## 1. 包含本地头文件和库目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/imgui
)

## 2. 查找外部库和工具链库
find_package(glfw3 REQUIRED)
message(STATUS "Found GLFW: ${glfw3_DIR}")
find_package(Vulkan REQUIRED)
message(STATUS "Found Vulkan: ${Vulkan_DIR}")
find_package(Ktx REQUIRED)
message(STATUS "Found Ktx: ${Ktx_DIR}")
find_package(glm REQUIRED)
message(STATUS "Found glm: ${glm_DIR}")
find_package(Imath REQUIRED)
message(STATUS "Found Imath: ${Imath_DIR}")
find_package(Alembic REQUIRED)
message(STATUS "Found Alembic: ${Alembic_DIR}")

## 3. 着色器和资产文件管理
find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator glslangValidator.exe)
if(NOT GLSLANG_VALIDATOR_EXECUTABLE)
    message(FATAL_ERROR "glslangValidator not found. Please ensure it's in your PATH.")
endif()

# 设置着色器源文件目录
set(SHADERS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/shaders)
set(SHADER_DEPENDENCY_FILES
    ${SHADERS_SOURCE_DIR}/simulation_header.glsl
    ${SHADERS_SOURCE_DIR}/render_header.glsl
)

# 寻找所有着色器源文件 (.glsl)
file(GLOB SHADER_GLSL_FILES
    ${SHADERS_SOURCE_DIR}/*.glsl
)

# 寻找所有着色器编译文件 (.comp, .vert, .frag)
file(GLOB SHADER_COMPILE_FILES
    ${SHADERS_SOURCE_DIR}/*.comp
    ${SHADERS_SOURCE_DIR}/*.vert
    ${SHADERS_SOURCE_DIR}/*.frag
)

# 创建一个列表来存储编译后的着色器文件
set(COMPILED_SHADERS)

# 遍历每个着色器编译文件并为其创建自定义命令
foreach(SHADER_FILE IN LISTS SHADER_COMPILE_FILES)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)
    string(REPLACE "." "" SHADER_EXT_NO_DOT ${SHADER_EXT})

    set(OUTPUT_SPV_FILE "${SHADERS_SOURCE_DIR}/${SHADER_NAME}_${SHADER_EXT_NO_DOT}.spv")

    add_custom_command(
        OUTPUT ${OUTPUT_SPV_FILE}
        COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} -g0 -gVS -V --target-env vulkan1.1 -o "${OUTPUT_SPV_FILE}" "${SHADER_FILE}"
        DEPENDS
            ${SHADER_FILE}
            ${SHADER_DEPENDENCY_FILES}
        COMMENT "Compiling shader ${SHADER_FILE}"
    )

    list(APPEND COMPILED_SHADERS ${OUTPUT_SPV_FILE})
endforeach()

# 创建着色器编译的自定义目标，并将其从解决方案中隐藏
add_custom_target(shaders_target ALL DEPENDS ${COMPILED_SHADERS})

## 4. 添加源文件和构建可执行文件
set(IMGUI_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui.cpp
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui_draw.cpp
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui_widgets.cpp
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui_tables.cpp
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui_demo.cpp
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui_impl_glfw.cpp
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui_impl_vulkan.cpp
)
set(SOURCE_FILES
    main.cpp
    LBM.cpp
    ${IMGUI_SOURCE_FILES}
)

# 寻找所有资产文件
file(GLOB TEXTURE_ASSETS "${PROJECT_SOURCE_DIR}/textures/*")
file(GLOB MODEL_ASSETS "${PROJECT_SOURCE_DIR}/models/*")
file(GLOB STL_ASSETS "${PROJECT_SOURCE_DIR}/stl/*")

# **将所有源文件和资产文件添加到可执行文件**
list(APPEND SOURCE_FILES
    ${SHADER_GLSL_FILES}
    ${SHADER_COMPILE_FILES}
    ${TEXTURE_ASSETS}
    ${MODEL_ASSETS}
    ${STL_ASSETS}
)

# 明确地为 .obj 文件设置属性，防止链接器错误
set_property(SOURCE ${MODEL_ASSETS} PROPERTY HEADER_FILE_ONLY ON)

# 添加源代码和非编译文件
add_executable(LBM_Free_Surface ${SOURCE_FILES})

# 将着色器目标作为依赖添加到可执行文件
add_dependencies(LBM_Free_Surface shaders_target)

# 使用 source_group 和 REGULAR_EXPRESSION 正确地创建虚拟文件夹
source_group("Shaders" REGULAR_EXPRESSION "${PROJECT_SOURCE_DIR}/shaders/.*")
source_group("Models" REGULAR_EXPRESSION "${PROJECT_SOURCE_DIR}/models/.*")
source_group("Textures" REGULAR_EXPRESSION "${PROJECT_SOURCE_DIR}/textures/.*")
source_group("STL" REGULAR_EXPRESSION "${PROJECT_SOURCE_DIR}/stl/.*")

# 链接库
target_link_libraries(LBM_Free_Surface PRIVATE
    glfw
    glm::glm
    Vulkan::Vulkan
    KTX::ktx
    Imath::Imath
    Alembic::Alembic
)

# 设置运行时的工作目录
set_property(TARGET LBM_Free_Surface PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_property(TARGET LBM_Free_Surface PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")