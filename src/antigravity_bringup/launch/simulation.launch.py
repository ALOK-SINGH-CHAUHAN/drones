"""
ANTIGRAVITY Bringup — Simulation Launch (Full Stack)
=====================================================
Launches the complete ANTIGRAVITY stack in Gazebo SITL mode.
Uses simulated sensors. Includes all layers up through planning + safety.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ─── Launch Arguments ────────────────────────────────────────────────
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz2 for visualization'
    )
    world_file_arg = DeclareLaunchArgument(
        'world_file', default_value='indoor_corridor.world',
        description='Gazebo world file'
    )
    map_file_arg = DeclareLaunchArgument(
        'map_file', default_value='',
        description='Path to pre-built map file'
    )
    use_gpu_arg = DeclareLaunchArgument(
        'use_gpu', default_value='true',
        description='Use GPU for inference (disable for CI testing)'
    )
    use_rl_arg = DeclareLaunchArgument(
        'use_rl', default_value='false',
        description='Enable RL decision layer'
    )

    bringup = FindPackageShare('antigravity_bringup')

    # Config paths
    slam_cfg = PathJoinSubstitution([bringup, 'config', 'slam.yaml'])
    detection_cfg = PathJoinSubstitution([bringup, 'config', 'detection.yaml'])
    localization_cfg = PathJoinSubstitution([bringup, 'config', 'localization.yaml'])
    cognition_cfg = PathJoinSubstitution([bringup, 'config', 'cognition.yaml'])
    planning_cfg = PathJoinSubstitution([bringup, 'config', 'planning.yaml'])
    control_cfg = PathJoinSubstitution([bringup, 'config', 'control.yaml'])
    ekf_cfg = PathJoinSubstitution([bringup, 'config', 'ekf.yaml'])
    safety_cfg = PathJoinSubstitution([bringup, 'config', 'safety.yaml'])

    # ─── Gazebo Simulation ───────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             LaunchConfiguration('world_file')],
        output='screen',
    )

    # ─── PX4 SITL ────────────────────────────────────────────────────────
    px4_sitl = ExecuteProcess(
        cmd=['px4', '-s', 'etc/init.d-posix/rcS', '-i', '0'],
        output='screen',
    )

    # ─── Spawn Drone Model ──────────────────────────────────────────────
    spawn_drone = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=[
            '-entity', 'antigravity_drone',
            '-topic', '/robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '0.5',
        ],
        output='screen',
    )

    # ─── Simulated SLAM Node ────────────────────────────────────────────
    slam_node = Node(
        package='antigravity_slam', executable='orb_slam3_node',
        name='orb_slam3', namespace='slam',
        parameters=[slam_cfg],
        remappings=[
            ('camera/image_raw', '/drone/camera/image_raw'),
            ('camera/depth', '/drone/camera/depth'),
            ('imu/data', '/drone/imu/data'),
        ],
        output='screen',
    )

    # ─── Detection Node ─────────────────────────────────────────────────
    detection_node = Node(
        package='antigravity_detection', executable='yolo_detection_node',
        name='yolo_detector', namespace='detection',
        parameters=[detection_cfg],
        remappings=[('camera/image_raw', '/drone/camera/image_raw')],
        output='screen',
    )

    # ─── Map Server ─────────────────────────────────────────────────────
    map_server_node = Node(
        package='antigravity_mapping', executable='map_server_node',
        name='map_server', namespace='mapping',
        parameters=[{'map_file': LaunchConfiguration('map_file')}],
        output='screen',
    )

    # ─── Localization ────────────────────────────────────────────────────
    localization_node = Node(
        package='antigravity_localization', executable='mcl_node',
        name='mcl_localization', namespace='localization',
        parameters=[localization_cfg],
        output='screen',
    )

    # ─── Cognition Nodes ─────────────────────────────────────────────────
    cognition_group = GroupAction([
        PushRosNamespace('cognition'),
        Node(
            package='antigravity_cognition', executable='octomap_world_model_node',
            name='world_model', parameters=[cognition_cfg], output='screen',
        ),
        Node(
            package='antigravity_cognition', executable='semantic_segmentation_node',
            name='semantic_seg', parameters=[cognition_cfg], output='screen',
        ),
        Node(
            package='antigravity_cognition', executable='prediction_engine_node',
            name='prediction', parameters=[cognition_cfg], output='screen',
        ),
    ])

    # ─── Planning Nodes ─────────────────────────────────────────────────
    planning_group = GroupAction([
        PushRosNamespace('planning'),
        Node(
            package='antigravity_planning', executable='global_planner_node',
            name='global_planner', parameters=[planning_cfg], output='screen',
        ),
        Node(
            package='antigravity_planning', executable='local_planner_node',
            name='local_planner', parameters=[planning_cfg], output='screen',
        ),
    ])

    rl_node = Node(
        package='antigravity_planning', executable='rl_decision_node',
        name='rl_decision', namespace='planning',
        parameters=[planning_cfg], output='screen',
        condition=IfCondition(LaunchConfiguration('use_rl')),
    )

    # ─── Control Nodes ──────────────────────────────────────────────────
    control_group = GroupAction([
        PushRosNamespace('control'),
        Node(
            package='antigravity_control', executable='px4_bridge_node',
            name='px4_bridge',
            parameters=[
                control_cfg,
                {'fcu_url': 'udp://:14540@127.0.0.1:14557'},
                {'simulation_mode': True},
            ],
            output='screen',
        ),
        Node(
            package='antigravity_control', executable='trajectory_optimizer_node',
            name='trajectory_optimizer', parameters=[ekf_cfg], output='screen',
        ),
        Node(
            package='antigravity_control', executable='ekf_state_estimator_node',
            name='ekf_estimator', parameters=[ekf_cfg], output='screen',
        ),
    ])

    # ─── Safety Nodes ───────────────────────────────────────────────────
    safety_group = GroupAction([
        PushRosNamespace('safety'),
        Node(
            package='antigravity_safety', executable='safety_arbiter_node',
            name='safety_arbiter', parameters=[safety_cfg], output='screen',
        ),
        Node(
            package='antigravity_safety', executable='geofence_node',
            name='geofence', parameters=[safety_cfg], output='screen',
        ),
        Node(
            package='antigravity_safety', executable='system_monitor_node',
            name='system_monitor', parameters=[safety_cfg], output='screen',
        ),
    ])

    # ─── RViz2 ───────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', PathJoinSubstitution([bringup, 'rviz', 'antigravity.rviz'])],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
    )

    # ─── Staged Launch ──────────────────────────────────────────────────
    stage_sim_startup = TimerAction(period=5.0, actions=[
        spawn_drone, slam_node, detection_node, map_server_node,
        localization_node,
    ])

    stage_intelligence = TimerAction(period=8.0, actions=[
        cognition_group, safety_group,
    ])

    stage_autonomy = TimerAction(period=10.0, actions=[
        planning_group, rl_node, control_group,
    ])

    return LaunchDescription([
        use_rviz_arg, world_file_arg, map_file_arg, use_gpu_arg, use_rl_arg,
        gazebo,
        px4_sitl,
        rviz_node,
        stage_sim_startup,
        stage_intelligence,
        stage_autonomy,
    ])
