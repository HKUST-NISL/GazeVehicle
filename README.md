# EyeVehicle
use eye gaze to control the movement of a vehicle in ros

### What's in this repo?
- [x] A robot with camera sensor
- [x] Face detection and analysis
- [x] Eye gaze estimation
- [x] Mouth status estimation
- [x] Control signal rendering to the robot


### Preparation
- Download the eye gaze [model](https://www.dropbox.com/sh/h23x33stlrhqvqq/AADn4iK7NMIc8bVnOkBpBBMSa?dl=0) and extract it to $ROOT_REPO
- Download other related asserts [asserts](https://www.dropbox.com/sh/pah5vjpvlohslzo/AABFl5nAcgtbosXDb9ZeqplWa?dl=0) and extract it to $ROOT_REPO


### Compile
```
cd $ROOT_REPO
catkin_make
echo '$ROOT_REPO/devel/setup.bash' > ~/.bashrc
source ~/.bashrc
```

### Demo

```
cd $ROOT_REPO
# show robot in gazebo
roslaunch mybot_gazebo mybot_world.launch

# show image from the robot camera
rosrun image_view image_view image:=/mybot/camera1/image_raw

# use mouth and eyes to send the comand
python script/eye_command_node.py
```

