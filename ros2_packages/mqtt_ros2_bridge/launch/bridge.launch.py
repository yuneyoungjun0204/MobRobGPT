"""
mqtt_ros2_bridge launch file
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    mqtt_host_arg = DeclareLaunchArgument(
        'mqtt_host',
        default_value='localhost',
        description='MQTT broker host'
    )

    mqtt_port_arg = DeclareLaunchArgument(
        'mqtt_port',
        default_value='9001',
        description='MQTT broker port'
    )

    n_allies_arg = DeclareLaunchArgument(
        'n_allies',
        default_value='3',
        description='Number of allied vessels'
    )

    n_enemies_arg = DeclareLaunchArgument(
        'n_enemies',
        default_value='10',
        description='Number of enemy vessels'
    )

    telemetry_rate_arg = DeclareLaunchArgument(
        'telemetry_rate',
        default_value='10.0',
        description='Telemetry publish rate (Hz)'
    )

    # Bridge node
    bridge_node = Node(
        package='mqtt_ros2_bridge',
        executable='bridge_node',
        name='mqtt_ros2_bridge',
        output='screen',
        parameters=[{
            'mqtt_host': LaunchConfiguration('mqtt_host'),
            'mqtt_port': LaunchConfiguration('mqtt_port'),
            'n_allies': LaunchConfiguration('n_allies'),
            'n_enemies': LaunchConfiguration('n_enemies'),
            'telemetry_rate': LaunchConfiguration('telemetry_rate'),
        }],
    )

    return LaunchDescription([
        mqtt_host_arg,
        mqtt_port_arg,
        n_allies_arg,
        n_enemies_arg,
        telemetry_rate_arg,
        bridge_node,
    ])
