from setuptools import setup

package_name = 'turtlebot3_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', ['launch/turtlebot3_launch.py']),
        (f'share/{package_name}/launch', ['launch/launch_sim.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chloe',
    maintainer_email='chloe@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = turtlebot3_controller.controller:main',
            'mapping = turtlebot3_controller.mapping:main',
            'explore = turtlebot3_controller.explore:main',
            'path = turtlebot3_controller.path:main',
        ],
    },
)
