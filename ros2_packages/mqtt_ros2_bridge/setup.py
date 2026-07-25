from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mqtt_ros2_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'paho-mqtt'],
    zip_safe=True,
    maintainer='yune',
    maintainer_email='yune@example.com',
    description='MQTT to ROS2 bridge for USV simulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge_node = mqtt_ros2_bridge.bridge_node:main',
            'usv_bridge = mqtt_ros2_bridge.usv_bridge:main',
        ],
    },
)
