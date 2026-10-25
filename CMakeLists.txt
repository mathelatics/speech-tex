cmake_minimum_required(VERSION 3.14)
project(phonos VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options — mirror ggml backend toggles so a single -D flag flips acceleration.
# Default policy: CPU everywhere; GPU backends opt-in per platform.
#   Android : -DPHONOS_GPU=OFF (CPU/ARM), later NNAPI/Vulkan via GGML_BACKEND_DL
#   macOS   : -DPHONOS_METAL=ON
#   Linux   : -DPHONOS_CUDA=ON or -DPHONOS_VULKAN=ON
#   Windows : same as Linux
# ---------------------------------------------------------------------------
option(PHONOS_BUILD_TESTS "phonos: build unit/integration tests"   ON)
option(PHONOS_BUILD_TOOLS "phonos: build command-line tools"       ON)
option(PHONOS_METAL       "phonos: enable Metal backend (Apple)"   OFF)
option(PHONOS_CUDA        "phonos: enable CUDA backend (NVIDIA)"   OFF)
option(PHONOS_VULKAN      "phonos: enable Vulkan backend"          OFF)
option(PHONOS_BLAS        "phonos: enable BLAS CPU acceleration"   OFF)
option(PHONOS_BACKEND_DL  "phonos: load GPU backends dynamically (single portable core lib)" OFF)

if (PHONOS_BACKEND_DL AND NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "shared libs required for dynamic backends" FORCE)
endif()

# ---------------------------------------------------------------------------
# Vendored ggml (MIT) — third_party/ggml, pinned in VENDOR_INFO.txt
# ---------------------------------------------------------------------------
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE       ON  CACHE BOOL "" FORCE)   # host-tuned SIMD (AVX2/AVX512 here, NEON on ARM)
set(GGML_OPENMP       ON  CACHE BOOL "" FORCE)
set(GGML_METAL        ${PHONOS_METAL}       CACHE BOOL "" FORCE)
set(GGML_CUDA         ${PHONOS_CUDA}        CACHE BOOL "" FORCE)
set(GGML_VULKAN       ${PHONOS_VULKAN}      CACHE BOOL "" FORCE)
set(GGML_BLAS         ${PHONOS_BLAS}        CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE    ON  CACHE BOOL "" FORCE)   # fast fused attention kernels on CPU
set(GGML_BACKEND_DL   ${PHONOS_BACKEND_DL}  CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml)

# ---------------------------------------------------------------------------
# Engine library: phonos_core (static by default; model logic + runtime glue)
# Layering rule: src/model & src/audio may ONLY include ggml.h / phonos headers.
# All device interaction funnels through src/runtime/backend_dispatch.cpp.
# ---------------------------------------------------------------------------
file(GLOB PHONOS_SOURCES CONFIGURE_DEPENDS
    src/core/*.c     src/core/*.cpp
    src/io/*.c       src/io/*.cpp
    src/model/*.c    src/model/*.cpp
    src/runtime/*.c  src/runtime/*.cpp
    src/audio/*.c    src/audio/*.cpp)

add_library(phonos_core STATIC ${PHONOS_SOURCES})
target_include_directories(phonos_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(phonos_core PUBLIC ggml)
if (MSVC)
    target_compile_options(phonos_core PRIVATE /W3)
else()
    target_compile_options(phonos_core PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Tools & tests
# ---------------------------------------------------------------------------
if (PHONOS_BUILD_TOOLS)
    add_executable(phonos-cli tools/phonos_cli.c)
    target_link_libraries(phonos-cli PRIVATE phonos_core)
endif()

if (PHONOS_BUILD_TESTS)
    enable_testing()
    add_executable(test_smoke tests/test_smoke.c)
    target_link_libraries(test_smoke PRIVATE phonos_core)
    add_test(NAME smoke COMMAND test_smoke)
endif()

message(STATUS "phonos: core lib = phonos_core, Metal=${PHONOS_METAL} CUDA=${PHONOS_CUDA} Vulkan=${PHONOS_VULKAN} BackendDL=${PHONOS_BACKEND_DL}")
