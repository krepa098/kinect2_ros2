cmake_minimum_required(VERSION 3.8)
project(kinect2_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_definitions(-DK2_CALIB_PATH="${PROJECT_SOURCE_DIR}/data/")

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(kinect2_registration REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(freenect2 REQUIRED)

add_executable(${PROJECT_NAME}_node
	src/kinect2_bridge.cpp
)

target_link_libraries(${PROJECT_NAME}_node freenect2)

include_directories(include/kinect2_bridge)

ament_target_dependencies(${PROJECT_NAME}_node 
                          rclcpp 
                          OpenCV 
                          kinect2_registration 
                          tf2 
                          tf2_ros 
                          tf2_geometry_msgs
                          sensor_msgs 
                          geometry_msgs)

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


install(
  TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY "include/"
  DESTINATION include
)

ament_export_include_directories(include/)
ament_package()

