# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import sys
import time
import sim as vrep # access all the VREP elements
import pathcontrol as pc
import sfla
import matplotlib.pyplot as plt


# %%
#<---------------------------------Initialization--------------------------------------->
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',-1,True,True,5000,5) # start a connection
if clientID!=-1:
	print ('Connected to remote API server')
else:
	print('Not connected to remote API server')
	sys.exit("No connection")

# Getting handles for the target
err, goal = vrep.simxGetObjectHandle(clientID, 'Goal', vrep.simx_opmode_blocking)

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
    ret, sensor_pos = vrep.simxGetObjectPosition(clientID, usensor[i], -1, vrep.simx_opmode_streaming)
    ret, sensor_orn = vrep.simxGetObjectOrientation(clientID, usensor[i], -1, vrep.simx_opmode_streaming)


# %%
#<-----------------------------------Control----------------------------------------->

ret, target = vrep.simxGetObjectPosition(clientID, goal, -1, vrep.simx_opmode_blocking)
target = np.array(target[:2])
ret, cur_pos = vrep.simxGetObjectPosition(clientID, robot, -1, vrep.simx_opmode_blocking)
cur_pos = np.array(cur_pos[:2])

# Create an instance of the solver
path_solver = sfla.sflaSolver(30, 5, 7, 12, 0.5)


# %%
path = np.empty((0,2))
sensed_obs = np.empty((0,2))
while np.linalg.norm(target - cur_pos) > 0.2:
    # Stop te wheels while we do another calculation
    errf = vrep.simxSetJointTargetVelocity(clientID, motorL, 0, vrep.simx_opmode_streaming)
    errf = vrep.simxSetJointTargetVelocity(clientID, motorR, 0, vrep.simx_opmode_streaming)
    # Read the input from the sensors
    obstacles = pc.sense_obstacles(clientID, usensor)
    # Get the new position in the path
    cur_pos, frogs, memeplexes = path_solver.sfla(cur_pos, target, obstacles)
    path = np.vstack((path, cur_pos))
    sensed_obs = np.vstack((sensed_obs, obstacles))
    errp = 10
    t = time.time()
    # Try to follow the given solution for 10 seconds
    while errp > 0.2 and time.time() - t < 10:
        avoid, ulb, urb = pc.braitenberg(clientID, usensor)
        errp, ulc, urc, pos, rot = pc.continuosControl(clientID, robot, cur_pos)
        # If an obstacle has to be avoided, give control to the braitenberg output
        ul = ulb if avoid else ulc
        ur = urb if avoid else urc
        errf = vrep.simxSetJointTargetVelocity(clientID, motorL, ul, vrep.simx_opmode_streaming)
        errf = vrep.simxSetJointTargetVelocity(clientID, motorR, ur, vrep.simx_opmode_streaming)


# The End
vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)


# %%
fig, ax = plt.subplots()

ax.plot(*path.T)
pc.imgscatter(ax, 'img/frog.png', path)

ax.scatter(*sensed_obs.T)
plt.show()


