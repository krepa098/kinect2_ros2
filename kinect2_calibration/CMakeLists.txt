cmake_minimum_required(VERSION 3.8)
project(kinect2_calibration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(compressed_image_transport REQUIRED)
find_package(compressed_depth_image_transport REQUIRED)
find_package(kinect2_bridge REQUIRED)
find_package(kinect2_registration REQUIRED)
find_package(OpenMP)

add_executable(${PROJECT_NAME}_node
	src/kinect2_calibration.cpp
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME}_node "${OpenMP_CXX_FLAGS}")
  target_compile_options(${PROJECT_NAME}_node INTERFACE "${OpenMP_CXX_FLAGS}")
endif()

include_directories(include)

ament_target_dependencies(${PROJECT_NAME}_node 
                          rclcpp 
                          OpenCV 
                          cv_bridge
                          sensor_msgs 
                          message_filters 
                          image_transport 
                          compressed_image_transport 
                          compressed_depth_image_transport
                          kinect2_bridge
                          kinect2_registration)

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

# install(DIRECTORY
#   launch
#   DESTINATION share/${PROJECT_NAME}/
# )

ament_package()

