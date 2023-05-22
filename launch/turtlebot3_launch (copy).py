import launch
import launch_ros.actions

def generate_launch_description():
    ld = launch.LaunchDescription()

    gazebo_cmd = launch.actions.ExecuteProcess(
        cmd=['ros2', 'launch', 'turtlebot3_gazebo', 'turtlebot3_dqn_stage4.launch.py'],
        output='screen'
    )
    ld.add_action(gazebo_cmd)

    delay_cmd = launch.actions.TimerAction(
        period=6.0,
        actions=[
            launch_ros.actions.Node(
                package='turtlebot3_controller',
                executable='controller',
                output='screen',
                name='controllernode'
            ),
            launch_ros.actions.Node(
                package='turtlebot3_controller',
                executable='mapping',
                output='screen',
                name='mappingnode'
            ),
            launch_ros.actions.Node(
                package='turtlebot3_controller',
                executable='explore',
                output='screen',
                name='explorenode'
            ),
            launch_ros.actions.Node(
                package='turtlebot3_controller',
                executable='path',
                output='screen',
                name='pathnode'
            )
        ]
    )
    ld.add_action(delay_cmd)

    return ld
