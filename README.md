# Perception2XSI

Perception2XSI is a driver for Softimage XSI used to connect Axis Neuron to Softimage XSI in realtime.

[![YOUTUBE](http://img.youtube.com/vi/nCxvTVdq5Hs/0.jpg)](http://www.youtube.com/watch?v=nCxvTVdq5Hs "YOUTUBE")

https://youtu.be/nCxvTVdq5Hs

## Features
- Realtime connection
- Recording in realtime
- Low delay

## Install
- Build and copy files into directory Users\Autodesk\Softimage_2014_SP2\Application\Plugins\Perception\bin\nt-x86-64
- don't forget to copy settingsPN.txt
- run command from XSI: PerceptionNeuron()
## in Axis Neuron settings:

- General:
  IP: yourIP (you should paste this IP into settingsPN.txt)
  Port: yourPORT(you should paste this IP into settingsPN.txt)
  
- Output Format:
  Quaternion: BVH local
  Acceleration: Local Sensor data
  Angular velocity:  Local Sensor data
  
- Broadcasting
  TCP/UDP: TCP
  Advanced BVH: disable
  BVH: enable, ServerPort: yourPORT(you should paste this IP into settingsPN.txt), Format: Binary, Use old header: enable
  Action: disable
  Calculation: enable, ServerPort: same as in General, format: binary
  
  
 

