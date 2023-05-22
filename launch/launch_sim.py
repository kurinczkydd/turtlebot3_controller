from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    ld = LaunchDescription()

    turtlebot3_gazebo_cmd = ExecuteProcess(
        cmd=['ros2', 'launch', 'turtlebot3_gazebo', 'turtlebot3_dqn_stage4.launch.py'],
        output='screen'
    )
    ld.add_action(turtlebot3_gazebo_cmd)

    rviz2_cmd = ExecuteProcess(
        cmd=['rviz2'],
        output='screen'
    )
    ld.add_action(rviz2_cmd)

    return ld

