from setuptools import find_packages, setup
package_name = 'antigravity_safety'
setup(
    name=package_name, version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='ANTIGRAVITY Dev', maintainer_email='dev@antigravity.ai',
    description='Safety arbiter and geofence enforcement',
    license='MIT',
    entry_points={'console_scripts': [
        'safety_arbiter_node = antigravity_safety.safety_arbiter_node:main',
        'geofence_node = antigravity_safety.geofence_node:main',
        'system_monitor_node = antigravity_safety.system_monitor_node:main',
    ]},
)
