# GazeVehicle
use eye gaze to control the movements of a vehicle in ros

<div align=center><img src="./asserts/demo.gif" width="800" height="450" ></div>

### What's in this repo?
- [x] A robot with camera sensor
- [x] Face detection and analysis
- [x] Eye gaze estimation
- [x] Mouth status estimation
- [x] Control signal rendering to the robot


### Preparation
- install tensorflow-gpu==1.14.0 (cuda-10.1, cudnn-7.6)
- install python libs: `dlib`, `scipy`
- Download the eye gaze [models](https://www.dropbox.com/sh/h23x33stlrhqvqq/AADn4iK7NMIc8bVnOkBpBBMSa?dl=0) and extract it to $ROOT_REPO


### Compile
```
cd $ROOT_REPO
catkin_make
```

### Demo

1. Show robot in gazebo simulator
<div align=center><img src="./asserts/bot_gazebo.png" width="800" height="450" ></div>

```
cd $ROOT_REPO
# show robot in gazebo
source ./devel/setup.bash
roslaunch mybot_gazebo mybot_world.launch
```

2. Show image obtained by the camera on the robot

<div align=center><img src="./asserts/bot_camera.png" width="800" height="450" ></div>

```
# show image from the robot camera
rosrun image_view image_view image:=/mybot/camera1/image_raw
```

3. Analyze mouth status and estimate eye gaze to control the robot

   - Eye gaze is to control the moving direction. 
      - look down to move backword
   - Start moving by open your mouth or push space key

```
# use mouth and eyes to send the comand
source ./devel/setup.bash
python scripts/eye_command_node.py
```

### Cite
This repo's model is from the paper below:
```
@inproceedings{chen2018appearance,
  title={Appearance-based gaze estimation using dilated-convolutions},
  author={Chen, Zhaokang and Shi, Bertram E},
  booktitle={Asian Conference on Computer Vision},
  pages={309--324},
  year={2018},
  organization={Springer}
}
```
