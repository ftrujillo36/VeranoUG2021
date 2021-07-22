import numpy as np
import time
import math as m
import sys
import sim as vrep # access all the VREP elements
import pathcontrol as pc

#<---------------------------------Initialization--------------------------------------->
vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)  # start a connection
if clientID!=-1:
	print ('Connected to remote API server')
else:
	print('Not connected to remote API server')
	sys.exit("No connection")

# Getting handles for the motors and robot
err, motorL = vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_blocking)
err, motorR = vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_blocking)
err, robot = vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx', vrep.simx_opmode_blocking)

# Assigning handles to the ultrasonic sensors
usensor = []
for i in range(1,17):
    err, s = vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx_ultrasonicSensor'+str(i), vrep.simx_opmode_blocking)
    usensor.append(s)

# Sensor initialization
for i in range(16):
    err, state, point, detectedObj, detectedSurfNormVec = vrep.simxReadProximitySensor(
        clientID, usensor[i], vrep.simx_opmode_streaming)

#<-----------------------------------Control----------------------------------------->
goals = [(3,5),(9,11),(14,15),(17,18)]
p = np.array(goals)

path = pc.splinePath(p[:,0], p[:,1])

pointsx = np.linspace(min(p[:,0]), max(p[:,0]), num=60, endpoint=True)
pointsy = path(pointsx)

step = 0
errp = 10
achieved = 0
avoid = False

while step < len(pointsx) and achieved < len(goals):
    prev = avoid
    # Traverse the path
    step = step + 1 if errp < 0.1 else step
    # Check obstacles or go to next point in path
    avoid, ulb, urb = pc.braitenberg(clientID, usensor)
    errp, ulc, urc, pos, rot = pc.continuosControl(clientID, robot, (pointsx[step], pointsy[step]))

    ul = ulb if avoid else ulc
    ur = urb if avoid else urc
    
    # Check achieved goals
    achieved = achieved + 1 if pc.distance2p(pos, goals[achieved]) <= 0.3 else achieved

    # If an obstacle was avoided, replan the path. Only works when there are more than 2 goals left
    if prev and not avoid:
        path  = pc.splinePath(p[:,0][achieved:], p[:,1][achieved:]) # New path of remaining points
        pointsx = np.linspace(min(p[:,0][achieved:]), max(p[:,0][achieved:]), num=(60 - step), endpoint=True)
        pointsy = path(pointsx)
        step = 0 # Start at the beginning of new path

    errf = vrep.simxSetJointTargetVelocity(clientID, motorL, ul, vrep.simx_opmode_streaming)
    errf = vrep.simxSetJointTargetVelocity(clientID, motorR, ur, vrep.simx_opmode_streaming)


# The End
vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
