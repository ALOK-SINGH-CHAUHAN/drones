from setuptools import find_packages, setup
package_name = 'antigravity_mapping'
setup(
    name=package_name, version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='ANTIGRAVITY Dev', maintainer_email='dev@antigravity.ai',
    description='Map server for occupancy grid and OctoMap',
    license='MIT',
    entry_points={'console_scripts': [
        'map_server_node = antigravity_mapping.map_server_node:main',
    ]},
)
