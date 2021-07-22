"""
Proyecto 1: Navegación reactiva con seguimiento de trayectoria
Funciones utilizadas
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math as m
import sim as vrep
#<---------------------------------Braitenberg--------------------------------------->

noDetectDist = 0.5
maxDist = 0.3
detectW = np.zeros(16)
braitenbergL = np.array([-0.2,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
braitenbergR = np.array([-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
v0 = 2
#obstacles = np.zeros((16,2))
#<-----------------------------------Control----------------------------------------->
# Controller gains (linear and heading)
Kv = 0.5
Kh = 2.5
hd = 0
r = 0.5*0.195
L = 0.311
#<----------------------------------Functions---------------------------------------->

def braitenberg(clientID, usensor):
    """
    Braitenberg algorithm for the front sensors of the pioneer 3dx
    """
    for i in range(len(usensor)):
        err, state, point, detectedObj, detectedSurfNormVec = vrep.simxReadProximitySensor(clientID, usensor[i], vrep.simx_opmode_buffer)
        distance = np.linalg.norm(point)
        # if a detection occurs
        if state and (distance < noDetectDist): # don't care about distant objects
            distance = max(distance, maxDist) 
            detectW[i] = 1 - ((distance - maxDist) / (noDetectDist - maxDist)) # Normalize the weight
        else:
            detectW[i] = 0
    dL = np.sum(braitenbergL * detectW)
    dR = np.sum(braitenbergR * detectW)
    vLeft = v0 + dL
    vRight = v0 + dR
    avoid = True if (abs(dL) + abs(dR)) else False
    return avoid, vLeft, vRight

def sense_obstacles(clientID, usensor):
    """
   
    """
    obstacles = np.empty((0,2))
    for i in range(len(usensor)):
        err, state, point, dObj, dSurfNormVec = vrep.simxReadProximitySensor(clientID, usensor[i], vrep.simx_opmode_buffer)
        # if a detection occurs
        if state:
            # Get the obstacle position
            ret, sensor_pos = vrep.simxGetObjectPosition(clientID, usensor[i], -1, vrep.simx_opmode_buffer)
            ret, sensor_orn = vrep.simxGetObjectOrientation(clientID, usensor[i], -1, vrep.simx_opmode_buffer)
            obstacles = np.vstack((obstacles, transformB2A(sensor_orn, sensor_pos, point)[:2]))

    return obstacles

def rotx(theta):
    Rx = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    return Rx

def roty(theta):
    Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    return Ry

def rotz(theta):
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    return Rz

def transformB2A(euler, trans, pos_b):
    """
    Transform point from system B to A given
    the translation and rotation angles
    """
    alpha = euler[0]
    beta = euler[1]
    gamma = euler[2]
    pos_b = np.array(pos_b)
    AB_R = np.dot(rotx(alpha), np.dot(roty(beta), rotz(gamma)))
    rotated = np.dot(AB_R, pos_b.reshape(3,1))
    return (rotated + np.array(trans).reshape(3,1)).reshape(3)

def mtransfB2A(rotAngle, B_ACoords):
    """
    Returns the 2d transformation matrix {B} to {A}
    """
    T = np.array([[np.cos(rotAngle) , np.sin(rotAngle), float(B_ACoords[0])],
                  [-np.sin(rotAngle), np.cos(rotAngle), float(B_ACoords[1])],
                  [                0,                0,           1]])
    return T

def splinePath(x, y):
    """
    Generate a function of the path that matches the goals
    """
    n = x.size
    if n >= 4:
        k = 'cubic'
    elif n == 3:
        k = 'quadratic'
    else:
        k = 'linear'
    f = interp1d(x, y, kind=k)
    return f

def angdiff(t1, t2):
    """
    Compute the angle difference, t2-t1, restricting the result to the [-pi,pi] range
    """
    # The angle magnitude comes from the dot product of two vectors
    angmag = m.acos(m.cos(t1)*m.cos(t2)+m.sin(t1)*m.sin(t2))
    # The direction of rotation comes from the sign of the cross product of two vectors
    angdir = m.cos(t1)*m.sin(t2)-m.sin(t1)*m.cos(t2)
    return m.copysign(angmag, angdir)

def continuosControl(clientID, robot, goal):
    """
    Provide control for the piooner 3dx given a goal
    """
    xd = goal[0]
    yd = goal[1]
    ret, pos = vrep.simxGetObjectPosition(clientID, robot, -1, vrep.simx_opmode_oneshot)
    ret, rot = vrep.simxGetObjectOrientation(clientID, robot, -1, vrep.simx_opmode_oneshot)

    errp = m.sqrt((xd-pos[0])**2 + (yd-pos[1])**2)
    angd = m.atan2(yd-pos[1], xd-pos[0])
    errh = angdiff(rot[2], angd)
    v = Kv*errp
    omega = Kh*errh
    ul = v/r - L*omega/(2*r)
    ur = v/r + L*omega/(2*r)
    return errp, ul, ur, pos, rot

def distance2p(a, b):
    """
    Returns the distance between 2 given points in R2
    """
    d = m.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    return d

def imgscatter(ax, img, points):
    """
    A scatter plot of the given points, using the image at the given path
    Arguments:
            ax - {axes.Axes} or array of Axea
            img - Path string
            points - numpy array of points to scatter
    """
    ax.scatter(*points.T)
    image = plt.imread(img)
    image_box = OffsetImage(image, zoom=0.03)
    for x0, y0 in points:
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        ax.add_artist(ab)