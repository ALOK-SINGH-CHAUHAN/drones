from setuptools import find_packages, setup

package_name = 'antigravity_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ANTIGRAVITY Dev',
    maintainer_email='dev@antigravity.ai',
    description='Camera and IMU driver integration for ANTIGRAVITY drone navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = antigravity_perception.camera_node:main',
            'imu_node = antigravity_perception.imu_node:main',
            'sensor_sync_node = antigravity_perception.sensor_sync_node:main',
            'sensor_diagnostics_node = antigravity_perception.sensor_diagnostics_node:main',
        ],
    },
)
