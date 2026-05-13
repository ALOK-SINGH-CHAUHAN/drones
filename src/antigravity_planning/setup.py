from setuptools import find_packages, setup
package_name = 'antigravity_planning'
setup(
    name=package_name, version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='ANTIGRAVITY Dev', maintainer_email='dev@antigravity.ai',
    description='Path planning for ANTIGRAVITY drone navigation',
    license='MIT',
    entry_points={'console_scripts': [
        'global_planner_node = antigravity_planning.global_planner_node:main',
        'local_planner_node = antigravity_planning.local_planner_node:main',
        'rl_decision_node = antigravity_planning.rl_decision_node:main',
    ]},
)
