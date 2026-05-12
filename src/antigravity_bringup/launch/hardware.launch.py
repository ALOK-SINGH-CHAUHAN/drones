"""
ANTIGRAVITY Bringup — Hardware Launch (Full Stack)
===================================================
Launches the full ANTIGRAVITY stack on physical hardware.
Includes all phases: Perception → SLAM → Detection → Cognition →
Planning → Control → Safety → Monitoring
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
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
    camera_type_arg = DeclareLaunchArgument(
        'camera_type', default_value='realsense',
        description='Camera type: realsense or zed'
    )
    map_file_arg = DeclareLaunchArgument(
        'map_file', default_value='',
        description='Path to pre-built map file (.yaml or .bt)'
    )
    use_slam_arg = DeclareLaunchArgument(
        'use_slam', default_value='true',
        description='Enable ORB-SLAM3'
    )
    use_detection_arg = DeclareLaunchArgument(
        'use_detection', default_value='true',
        description='Enable YOLOv8 detection'
    )
    use_localization_arg = DeclareLaunchArgument(
        'use_localization', default_value='true',
        description='Enable MCL particle filter localization'
    )
    use_cognition_arg = DeclareLaunchArgument(
        'use_cognition', default_value='true',
        description='Enable cognition layer (semantics + prediction)'
    )
    use_planning_arg = DeclareLaunchArgument(
        'use_planning', default_value='true',
        description='Enable planning layer (A* + MPC)'
    )
    use_safety_arg = DeclareLaunchArgument(
        'use_safety', default_value='true',
        description='Enable safety layer (arbiter + geofence)'
    )
    use_rl_arg = DeclareLaunchArgument(
        'use_rl', default_value='false',
        description='Enable RL decision layer'
    )
    px4_fcu_url_arg = DeclareLaunchArgument(
        'px4_fcu_url', default_value='/dev/ttyACM0:921600',
        description='PX4 FCU connection URL'
    )

    # ─── Configuration Paths ─────────────────────────────────────────────
    bringup = FindPackageShare('antigravity_bringup')
    perception_cfg = PathJoinSubstitution([bringup, 'config', 'perception.yaml'])
    slam_cfg = PathJoinSubstitution([bringup, 'config', 'slam.yaml'])
    detection_cfg = PathJoinSubstitution([bringup, 'config', 'detection.yaml'])
    control_cfg = PathJoinSubstitution([bringup, 'config', 'control.yaml'])
    localization_cfg = PathJoinSubstitution([bringup, 'config', 'localization.yaml'])
    cognition_cfg = PathJoinSubstitution([bringup, 'config', 'cognition.yaml'])
    planning_cfg = PathJoinSubstitution([bringup, 'config', 'planning.yaml'])
    ekf_cfg = PathJoinSubstitution([bringup, 'config', 'ekf.yaml'])
    safety_cfg = PathJoinSubstitution([bringup, 'config', 'safety.yaml'])

    # ═══ LAYER 1: PERCEPTION ═══════════════════════════════════════════
    perception_group = GroupAction([
        PushRosNamespace('perception'),
        Node(
            package='antigravity_perception', executable='camera_node',
            name='camera_driver', parameters=[perception_cfg],
            remappings=[
                ('camera/image_raw', '/camera/image_raw'),
                ('camera/depth', '/camera/depth'),
                ('camera/camera_info', '/camera/camera_info'),
            ],
            output='screen',
        ),
        Node(
            package='antigravity_perception', executable='imu_node',
            name='imu_driver', parameters=[perception_cfg],
            remappings=[('imu/data', '/imu/data')],
            output='screen',
        ),
        Node(
            package='antigravity_perception', executable='sensor_sync_node',
            name='sensor_synchronizer', parameters=[perception_cfg],
            output='screen',
        ),
    ])

    # ═══ LAYER 2: SLAM ═════════════════════════════════════════════════
    slam_node = Node(
        package='antigravity_slam', executable='orb_slam3_node',
        name='orb_slam3', namespace='slam',
        parameters=[slam_cfg],
        remappings=[
            ('camera/image_raw', '/camera/image_raw'),
            ('camera/depth', '/camera/depth'),
            ('imu/data', '/imu/data'),
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_slam')),
    )

    # ═══ LAYER 3: DETECTION ════════════════════════════════════════════
    detection_node = Node(
        package='antigravity_detection', executable='yolo_detection_node',
        name='yolo_detector', namespace='detection',
        parameters=[detection_cfg],
        remappings=[('camera/image_raw', '/camera/image_raw')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_detection')),
    )

    # ═══ LAYER 4: MAPPING ══════════════════════════════════════════════
    map_server_node = Node(
        package='antigravity_mapping', executable='map_server_node',
        name='map_server', namespace='mapping',
        parameters=[{'map_file': LaunchConfiguration('map_file')}],
        output='screen',
    )

    # ═══ LAYER 5: LOCALIZATION ═════════════════════════════════════════
    localization_node = Node(
        package='antigravity_localization', executable='mcl_node',
        name='mcl_localization', namespace='localization',
        parameters=[localization_cfg],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_localization')),
    )

    # ═══ LAYER 6: COGNITION ════════════════════════════════════════════
    cognition_group = GroupAction(
        actions=[
            PushRosNamespace('cognition'),
            Node(
                package='antigravity_cognition', executable='octomap_world_model_node',
                name='world_model', parameters=[cognition_cfg], output='screen',
            ),
            Node(
                package='antigravity_cognition', executable='semantic_segmentation_node',
                name='semantic_segmentation', parameters=[cognition_cfg], output='screen',
            ),
            Node(
                package='antigravity_cognition', executable='prediction_engine_node',
                name='prediction_engine', parameters=[cognition_cfg], output='screen',
            ),
        ],
        condition=IfCondition(LaunchConfiguration('use_cognition')),
    )

    # ═══ LAYER 7: PLANNING ═════════════════════════════════════════════
    planning_group = GroupAction(
        actions=[
            PushRosNamespace('planning'),
            Node(
                package='antigravity_planning', executable='global_planner_node',
                name='global_planner', parameters=[planning_cfg], output='screen',
            ),
            Node(
                package='antigravity_planning', executable='local_planner_node',
                name='local_planner', parameters=[planning_cfg], output='screen',
            ),
        ],
        condition=IfCondition(LaunchConfiguration('use_planning')),
    )

    rl_node = Node(
        package='antigravity_planning', executable='rl_decision_node',
        name='rl_decision', namespace='planning',
        parameters=[planning_cfg], output='screen',
        condition=IfCondition(LaunchConfiguration('use_rl')),
    )

    # ═══ LAYER 8: CONTROL ══════════════════════════════════════════════
    control_group = GroupAction([
        PushRosNamespace('control'),
        Node(
            package='antigravity_control', executable='px4_bridge_node',
            name='px4_bridge',
            parameters=[
                control_cfg,
                {'fcu_url': LaunchConfiguration('px4_fcu_url')},
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

    # ═══ LAYER 9: SAFETY ═══════════════════════════════════════════════
    safety_group = GroupAction(
        actions=[
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
        ],
        condition=IfCondition(LaunchConfiguration('use_safety')),
    )

    # ═══ VISUALIZATION ═════════════════════════════════════════════════
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', PathJoinSubstitution([bringup, 'rviz', 'antigravity.rviz'])],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
    )

    # ─── Staged Launch ──────────────────────────────────────────────────
    stage_2 = TimerAction(period=2.0, actions=[
        safety_group,
    ])

    stage_3 = TimerAction(period=3.0, actions=[
        cognition_group,
    ])

    stage_4 = TimerAction(period=5.0, actions=[
        planning_group, rl_node, control_group,
    ])

    return LaunchDescription([
        # Arguments
        use_rviz_arg, camera_type_arg, map_file_arg,
        use_slam_arg, use_detection_arg, use_localization_arg,
        use_cognition_arg, use_planning_arg, use_safety_arg,
        use_rl_arg, px4_fcu_url_arg,
        # Immediate
        perception_group,
        slam_node,
        detection_node,
        map_server_node,
        localization_node,
        rviz_node,
        # Staged
        stage_2,
        stage_3,
        stage_4,
    ])
