import launch
import launch_ros.actions
import launch.actions

def generate_launch_description():
    ld = launch.LaunchDescription()
    
    # Launch the Gazebo simulation
    #gazebo_cmd = launch_ros.actions.Node(
    #    package='turtlebot3_gazebo',
    #    executable='turtlebot3_dqn_stage4.launch.py',
    #    output='screen'
    #)
    #ld.add_action(gazebo_cmd)

    #gazebo_cmd = launch.actions.ExecuteProcess(
    #    cmd=['ros2', 'launch', 'turtlebot3_gazebo', 'turtlebot3_dqn_stage4.launch.py'],
    #    output='screen'
    #)
    #ld.add_action(gazebo_cmd)

    # Delayed reset command
    reset_cmd = launch.actions.TimerAction(
        period=0.5,
        actions=[
            launch.actions.ExecuteProcess(
                cmd=['ros2', 'service', 'call', '/reset_simulation', 'std_srvs/srv/Empty'],
                output='screen'
            )
        ]
    )
    ld.add_action(reset_cmd)
    
    delay_cmd = launch.actions.TimerAction(
        period=3.0,
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
            ),
            launch_ros.actions.Node(
                package='turtlebot3_controller',
                executable='sweep',
                output='screen',
                name='sweepnode'
            )
        ]
    )
    ld.add_action(delay_cmd)

    #controller_cmd = launch_ros.actions.Node(
    #    package='turtlebot3_controller',
    #    executable='controller',
    #    output='screen'
    #)
    #ld.add_action(controller_cmd)

    return ld
