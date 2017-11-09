cmake_minimum_required (VERSION 3.1)

project(piopt)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -Wall -mfpu=neon")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -mfpu=neon")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mfpu=neon")

# The version number.
set(piorb_VERSION_MAJOR 0)
set(piorb_VERSION_MINOR 1)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
 
find_package(Eigen3 3.1.0 REQUIRED)
find_package(GTest REQUIRED)

include_directories(
  "${PROJECT_SOURCE_DIR}/include"
  "${EIGEN3_INCLUDE_DIR}"
  "${GTEST_INCLUDE_DIRS}"
  )
 
add_library(TestUtil SHARED
  test/TestUtil.cpp
  )

add_executable(GaussianTest
  test/GaussianTest.cpp
  )
target_link_libraries(GaussianTest TestUtil ${GTEST_BOTH_LIBRARIES})

add_executable(BilinearTest
  test/BilinearTest.cpp
  )
target_link_libraries(BilinearTest TestUtil ${GTEST_BOTH_LIBRARIES})

add_executable(FeatureGridTest
  test/FeatureGridTest.cpp
  )
target_link_libraries(FeatureGridTest TestUtil ${GTEST_BOTH_LIBRARIES})

