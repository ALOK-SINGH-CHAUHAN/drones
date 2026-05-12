from setuptools import find_packages, setup
package_name = 'antigravity_control'
setup(
    name=package_name, version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='ANTIGRAVITY Dev', maintainer_email='dev@antigravity.ai',
    description='PX4 flight control integration',
    license='MIT',
    entry_points={'console_scripts': [
        'px4_bridge_node = antigravity_control.px4_bridge_node:main',
        'trajectory_optimizer_node = antigravity_control.trajectory_optimizer_node:main',
        'ekf_state_estimator_node = antigravity_control.ekf_state_estimator_node:main',
    ]},
)
