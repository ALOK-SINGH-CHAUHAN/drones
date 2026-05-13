"""
ANTIGRAVITY Bringup — Full Stack Launch
========================================
Launches the COMPLETE ANTIGRAVITY autonomous navigation stack.
Includes all phases: Perception → SLAM → Detection → Cognition →
Planning → Control → Safety → Monitoring

This is the production launch file for autonomous flight.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ─── Launch Arguments ────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='Launch RViz2 visualization'),
        DeclareLaunchArgument('camera_type', default_value='realsense',
                              description='Camera: realsense or zed'),
        DeclareLaunchArgument('map_file', default_value='',
                              description='Pre-built map file (.yaml or .bt)'),
        DeclareLaunchArgument('use_slam', default_value='true',
                              description='Enable ORB-SLAM3'),
        DeclareLaunchArgument('use_detection', default_value='true',
                              description='Enable YOLOv8 detection'),
        DeclareLaunchArgument('use_localization', default_value='true',
                              description='Enable MCL localization'),
        DeclareLaunchArgument('use_cognition', default_value='true',
                              description='Enable cognition layer'),
        DeclareLaunchArgument('use_planning', default_value='true',
                              description='Enable planning layer'),
        DeclareLaunchArgument('use_safety', default_value='true',
                              description='Enable safety layer'),
        DeclareLaunchArgument('use_rl', default_value='false',
                              description='Enable RL decision layer'),
        DeclareLaunchArgument('simulation_mode', default_value='false',
                              description='Run in simulation mode'),
        DeclareLaunchArgument('px4_fcu_url', default_value='/dev/ttyACM0:921600',
                              description='PX4 FCU connection URL'),
    ]

    bringup = FindPackageShare('antigravity_bringup')

    # ─── Config Paths ────────────────────────────────────────────────────
    perception_cfg = PathJoinSubstitution([bringup, 'config', 'perception.yaml'])
    slam_cfg = PathJoinSubstitution([bringup, 'config', 'slam.yaml'])
    detection_cfg = PathJoinSubstitution([bringup, 'config', 'detection.yaml'])
    localization_cfg = PathJoinSubstitution([bringup, 'config', 'localization.yaml'])
    cognition_cfg = PathJoinSubstitution([bringup, 'config', 'cognition.yaml'])
    planning_cfg = PathJoinSubstitution([bringup, 'config', 'planning.yaml'])
    control_cfg = PathJoinSubstitution([bringup, 'config', 'control.yaml'])
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
            name='sensor_sync', parameters=[perception_cfg],
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
                name='semantic_seg', parameters=[cognition_cfg], output='screen',
            ),
            Node(
                package='antigravity_cognition', executable='prediction_engine_node',
                name='prediction', parameters=[cognition_cfg], output='screen',
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

    # ═══ LAYER 7b: RL DECISION (optional) ══════════════════════════════
    rl_node = Node(
        package='antigravity_planning', executable='rl_decision_node',
        name='rl_decision', namespace='planning',
        parameters=[planning_cfg],
        output='screen',
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
                {'simulation_mode': LaunchConfiguration('simulation_mode')},
            ],
            output='screen',
        ),
        Node(
            package='antigravity_control', executable='trajectory_optimizer_node',
            name='trajectory_optimizer',
            parameters=[ekf_cfg],
            output='screen',
        ),
        Node(
            package='antigravity_control', executable='ekf_state_estimator_node',
            name='ekf_estimator',
            parameters=[ekf_cfg],
            output='screen',
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

    # ─── Staged Launch Order ─────────────────────────────────────────
    # Perception + Mapping start immediately
    # SLAM, Detection, Localization at T+1s
    # Cognition at T+3s (needs perception data)
    # Planning + Control at T+5s (needs cognition + map)
    # Safety at T+2s (needs to be up early)

    stage_1 = TimerAction(period=1.0, actions=[
        slam_node, detection_node, localization_node,
    ])

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
        *args,
        # Immediate launch
        perception_group,
        map_server_node,
        rviz_node,
        # Staged launches
        stage_1,
        stage_2,
        stage_3,
        stage_4,
    ])
