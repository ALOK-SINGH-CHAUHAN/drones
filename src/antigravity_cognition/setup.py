from setuptools import find_packages, setup
package_name = 'antigravity_cognition'
setup(
    name=package_name, version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='ANTIGRAVITY Dev', maintainer_email='dev@antigravity.ai',
    description='Cognition layer for world modeling, semantics, and prediction',
    license='MIT',
    entry_points={'console_scripts': [
        'octomap_world_model_node = antigravity_cognition.octomap_world_model_node:main',
        'semantic_segmentation_node = antigravity_cognition.semantic_segmentation_node:main',
        'prediction_engine_node = antigravity_cognition.prediction_engine_node:main',
    ]},
)
