from setuptools import find_packages, setup
package_name = 'antigravity_localization'
setup(
    name=package_name, version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='ANTIGRAVITY Dev', maintainer_email='dev@antigravity.ai',
    description='Monte Carlo Localization for drone navigation',
    license='MIT',
    entry_points={'console_scripts': [
        'mcl_node = antigravity_localization.mcl_node:main',
    ]},
)
