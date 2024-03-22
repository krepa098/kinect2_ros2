from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode

def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration('name').perform(context)

    parameters = [
        {
            "frame_id": name+"_link",
            "subscribe_rgb": True,
            "subscribe_depth": True,
            "subscribe_odom_info": False,
            "approx_sync": False,
            "Rtabmap/DetectionRate": "3.5",
            "subscribe_scan_cloud": True,
            "set_mode_mapping": True,
        }
    ]

    remappings = [
        ("rgb/image", name+"/qhd/image_color_rect"),
        ("rgb/camera_info", name+"/qhd/camera_info"),
        ("depth/image", name+"/qhd/image_depth_rect"),
        ("scan_cloud", name+"/qhd/points"),
    ]

    return [
       Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            output="screen",
            remappings=remappings,
            parameters=parameters,
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            output="screen",
            arguments = ['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0', '--frame-id', 'base_link', '--child-frame-id', name+'_link']
        ),

        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            output="screen",
            remappings=remappings,
            parameters=parameters,
        ),

        Node(
            package="rtabmap_viz",
            executable="rtabmap_viz",
            output="screen",
            parameters=parameters,
            remappings=remappings,
        ),
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("name", default_value="kinect2"),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
