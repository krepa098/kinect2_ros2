cmake_minimum_required(VERSION 3.9)
project(kinect2_registration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3)
find_package(OpenMP)
# find_package(OpenCL)

# CPU based depth registration
if(EIGEN3_FOUND)
  message(STATUS "CPU based depth registration enabled")
  include_directories(${EIGEN3_INCLUDE_DIR})
  set(DEPTH_REG_CPU ON)
  add_definitions(-DDEPTH_REG_CPU)
else()
  message(STATUS "CPU based depth registration disabled")
  set(DEPTH_REG_CPU OFF)
endif()

# OpenCL based depth registration
# if(OpenCL_FOUND)
#   message(STATUS "OpenCL based depth registration enabled")
#   set(EXPORTED_DEPENDENCIES OpenCL)
#   include_directories(${OpenCL_INCLUDE_DIRS})

#   set(DEPTH_REG_OPENCL ON)
#   add_definitions(-DDEPTH_REG_OPENCL)
# else()
#   message(STATUS "OpenCL based depth registration disabled")
#   set(DEPTH_REG_OPENCL OFF)
# endif()

add_library(${PROJECT_NAME}
	src/depth_registration_cpu.cpp
	src/kinect2_registration.cpp
	# src/depth_registration_opencl.cpp
)

target_include_directories(${PROJECT_NAME} 
                          PRIVATE 
                          src/ include/internal 
                          PUBLIC 
                          include/)

if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} "${OpenMP_CXX_FLAGS}")
  target_compile_options(${PROJECT_NAME} INTERFACE "${OpenMP_CXX_FLAGS}")
endif()

ament_target_dependencies(${PROJECT_NAME} rclcpp OpenCV Eigen3)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# install header files
install(DIRECTORY include/${PROJECT_NAME}/
  	DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)

ament_export_include_directories(include/)
ament_export_libraries(${PROJECT_NAME})
ament_package()

