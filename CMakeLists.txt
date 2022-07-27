cmake_minimum_required(VERSION 3.1.3)
project(pynao)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    rospy
    std_msgs
    nao_moveit_config
)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread)


# Generate messages in the 'msg' folder
 add_message_files(
   FILES
   Num.msg
 )

# Generate services in the 'srv' folder
 add_service_files(
   FILES
   AddTwoInts.srv
 )

# Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
 )

catkin_package(
  LIBRARIES
  INCLUDE_DIRS
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    message_runtime
  DEPENDS
    EIGEN3
)


###########
## Build ##
###########

include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})

#############
## Install ##
#############

# Mark executable scripts (Python etc.) for installation
# in contrast to setup.py, you can choose the destination
 catkin_install_python(PROGRAMS
   scripts/poc.py
   scripts/sender.py
   scripts/receiver.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
