cmake_minimum_required(VERSION 2.8)
project(car_counter)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( car_counter src/car_counter.cpp )
target_link_libraries( car_counter ${OpenCV_LIBS} )
# Copy static files
file(COPY src/video_1.mov DESTINATION ${CMAKE_BINARY_DIR})