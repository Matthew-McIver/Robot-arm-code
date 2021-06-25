#!/usr/bin/env python3

import vrep as sim
import time
import cv2
import numpy as np
from sympy import Matrix, pi, sin, cos, symbols, diff, solve





def T(x, y, z):
    T_xyz = Matrix([[1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])
    return T_xyz

def Rx(roll):
    R_x = Matrix([[1, 0, 0, 0],
                  [0, cos(roll), -sin(roll), 0],
                  [0, sin(roll), cos(roll), 0],
                  [0, 0, 0, 1]])
    return R_x

def Ry(pitch):
    R_y = Matrix([[cos(pitch),  0, sin(pitch), 0],
                  [0,           1, 0,          0],
                  [-sin(pitch), 0, cos(pitch), 0],
                  [0,           0, 0,          1]])
    return R_y

def Rz(yaw):
    R_z = Matrix([[cos(yaw), -sin(yaw), 0, 0],
                  [sin(yaw), cos(yaw), 0, 0 ],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return R_z

# Declare Symbols
theta1,theta2,theta3,theta4,theta5,theta6 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6')
theta = Matrix([theta1,theta2,theta3,theta4,theta5,theta6])
dtheta1,dtheta2,dtheta3,dtheta4,dtheta5,dtheta6 = symbols('dtheta_1 dtheta_2 dtheta_3 dtheta_4 dtheta_5 dtheta_6')
dtheta = Matrix([dtheta1,dtheta2,dtheta3,dtheta4,dtheta5,dtheta6])

# Fixed lengths
T1 = Ry(-pi/2) * T(0.065, 0, 0) * Rx(theta1 - pi/2)
T2 = T1 * T(0.139, 0, 0) * Rz(-theta2)
T3 = T2 * T(0.096, 0, 0) * Rx(theta3)
T4 = T3 * T(0.195, 0, 0) * Rz(-theta4 + pi/2)
T5 = T4 * T(0.167-0.076, 0, 0) * Rx(theta5 + pi)
T6 = T5 * T(0.104+0.129, 0, 0) * Rz(theta6 + pi/2)

T7 = T6 * T(0.255, 0, 0)

p0 = Matrix([0,0,0,1])
p1 = T1 * p0
p2 = T2 * p0
p3 = T3 * p0
p4 = T4 * p0
p5 = T5 * p0
p6 = T6 * p0

p7 = T7 * p0


def inverse_kinematic(xyz):
    # Get joint angles
    joint1Orientation = sim.simxGetJointPosition(clientID, joint1, sim.simx_opmode_oneshot)[1]
    joint2Orientation = sim.simxGetJointPosition(clientID, joint2, sim.simx_opmode_oneshot)[1]
    joint3Orientation = sim.simxGetJointPosition(clientID, joint3, sim.simx_opmode_oneshot)[1]
    joint4Orientation = sim.simxGetJointPosition(clientID, joint4, sim.simx_opmode_oneshot)[1]
    joint5Orientation = sim.simxGetJointPosition(clientID, joint5, sim.simx_opmode_oneshot)[1]
    joint6Orientation = sim.simxGetJointPosition(clientID, joint6, sim.simx_opmode_oneshot)[1]

    joint_angles = [joint1Orientation,joint2Orientation,joint3Orientation,joint4Orientation,joint5Orientation,joint6Orientation]
    theta_i = Matrix(joint_angles)

    p_i = Matrix([p6[0], p6[1], p6[2]])

    # Matrix of destination coordinates
    p_f = Matrix(xyz)

    # Define Jacobian
    J = p_i.jacobian(theta)

    # Calculate Movement Vector
    dp = p_f - p_i.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]})

    n=10
    for x in range(n):
        Jsub = J.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]})
        Jeval = Jsub.evalf()

        sol = solve(Jeval*dtheta-dp/n,(dtheta1,dtheta2,dtheta3,dtheta4,dtheta5,dtheta6), particular=True, quick=True, set=True)

        (element,) = sol[1]
        while (len(element) < 6):
            element += (0,)
        
        theta_i += Matrix(element)
    
    # Make Movement
    return theta_i

def look_down():
    # Get joint angles
    joint1Orientation = sim.simxGetJointPosition(clientID, joint1, sim.simx_opmode_oneshot)[1]
    joint2Orientation = sim.simxGetJointPosition(clientID, joint2, sim.simx_opmode_oneshot)[1]
    joint3Orientation = sim.simxGetJointPosition(clientID, joint3, sim.simx_opmode_oneshot)[1]
    joint4Orientation = sim.simxGetJointPosition(clientID, joint4, sim.simx_opmode_oneshot)[1]
    joint5Orientation = sim.simxGetJointPosition(clientID, joint5, sim.simx_opmode_oneshot)[1]
    joint6Orientation = sim.simxGetJointPosition(clientID, joint6, sim.simx_opmode_oneshot)[1]

    joint_angles = [joint1Orientation,joint2Orientation,joint3Orientation,joint4Orientation,joint5Orientation,joint6Orientation]
    theta_i = Matrix(joint_angles)

    p_i = Matrix([p7[0], p7[1], p7[2]])

    dp = Matrix([ 0.255,0,0])

    p_f = p_i + dp

    # Matrix of destination coordinates
    p_f = Matrix(p_f)

    # Define Jacobian
    J = p_i.jacobian(theta)

    n=10
    for x in range(n):
        Jsub = J.subs({theta1:theta_i[0], theta2:theta_i[1], theta3:theta_i[2], theta4:theta_i[3], theta5:theta_i[4], theta6:theta_i[5]})
        Jeval = Jsub.evalf()

        sol = solve(Jeval*dtheta-dp/n,(dtheta1,dtheta2,dtheta3,dtheta4,dtheta5,dtheta6), particular=True, quick=True, set=True)

        (element,) = sol[1]
        while (len(element) < 6):
            element += (0,)
        
        theta_i += Matrix(element)
    
    # Make Movement
    return theta_i

flag = 0

#Program Variables
joint1Angle = 0.0
joint2Angle = 0.0
joint3Angle = 0.0
joint4Angle = 0.0
joint5Angle = pi
joint6Angle = 0.0

#Start Program and just in case, close all opened connections
print('Program started')
sim.simxFinish(-1)

#Connect to simulator running on localhost
#V-REP runs on port 19997, a script opens the API on port 19999
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
thistime = time.time()

#Connect to the simulation
if clientID != -1:
    print('Connected to remote API server')

    #Get handles to simulation objects
    print('Obtaining handles of simulation objects')
    #Floor
    res,floor = sim.simxGetObjectHandle(clientID, 'ResizableFloor_5_25', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Floor')
    #Robot Arm Base (reference point)
    res,arm = sim.simxGetObjectHandle(clientID, 'redundantRobot', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Robot')
    #End Effector camera for visual servoing
    res,camera = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to Camera')
    #Joint 1
    res,joint1 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint1', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 2
    res,joint2 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint2', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 3
    res,joint3 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint3', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 4
    res,joint4 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint4', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 5
    res,joint5 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint5', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    #Joint 6
    res,joint6 = sim.simxGetObjectHandle(clientID, 'redundantRob_joint6', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok: print('Could not get handle to collisionSensor')
    

    #Start main control loop
    print('Starting control loop')
    res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera, 0, sim.simx_opmode_streaming)
    while (sim.simxGetConnectionId(clientID) != -1):
        #Get image from Camera
        lasttime = thistime
        thistime = time.time()
        res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera, 0, sim.simx_opmode_buffer)
        if res == sim.simx_return_ok:
            #Process image 
            # print("Image OK!", "{:02.1f}".format(1.0 / (thistime - lasttime)), "FPS")
            #Convert from V-REP flat RGB representation to OpenCV BGR colour planes
            original = np.array(image, dtype=np.uint8)
            original.resize([resolution[0], resolution[1], 3])
            original = cv2.flip(original, 0)
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            #Remove distortion from image
            ratio = resolution[0] / resolution[1]
            #cv2.initCameraMatrix2D(None, None)
            #cv2.calibrateCamera()
            #Filter the image into components (alternately, cv2.split includes components of greys, etc.)
            blue = cv2.inRange(original, np.array([224,0,0]), np.array([255,32,32]))
            green = cv2.inRange(original, np.array([0,224,0]), np.array([32,255,32]))
            red = cv2.inRange(original, np.array([0,0,224]), np.array([32,32,255]))
            #Apply Canny edge detection
            blueEdges = cv2.Canny(blue, 32, 64)
            greenEdges = cv2.Canny(green, 32, 64)
            redEdges = cv2.Canny(red, 32, 64)
            #Combine edges from red, green, and blue channels
            edges = cv2.merge((blueEdges, greenEdges, redEdges))
            
            # calculate moments of binary image
            M_r = cv2.moments(red)
            M_g = cv2.moments(green)
            M_b = cv2.moments(blue)
            # calculate x,y coordinate of center
            try:

                cX_g = int(M_g["m10"] / M_g["m00"])
                cY_g = int(M_g["m01"] / M_g["m00"])

                # put text and highlight the center
                cv2.circle(edges, (cX_g, cY_g), 5, (255, 255, 255), -1)
                cv2.putText(edges, "centroid", (cX_g - 25, cY_g - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except:
                pass
            
            #Images must all be the same dimensions as reported by original.shape 
            images = np.vstack((original, edges))
            #Show processed images together in OpenCV window
            cv2.imshow('Camera', images)
            components = np.vstack((blue, green, red))
            cv2.imshow('Components', components)


        elif res == sim.simx_return_novalue_flag:
            #Camera has not started or is not returning images
            print("No image yet")
            pass

        else:
            #Something else has happened
            print("Unexpected error returned", res)

        
        #Calculate your Forward and Inverse Kinematics here
        coords = [
            [0, 0.25, 0.5],
            [0.25, 0.25, 0.5],
            [0, -0.5, 0.5]
        ]
        

        theta_i = inverse_kinematic(coords[flag])

        joint1Angle = theta_i[0]
        joint2Angle = theta_i[1]
        joint3Angle = theta_i[2]
        joint4Angle = theta_i[3]

        flag += 1
        if flag > len(coords) - 1:
            flag = 0



        #Place your Mobile Robot Control code here

        #Set actuators on mobile robot
        sim.simxSetJointTargetPosition(clientID, joint1, joint1Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint2, joint2Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint3, joint3Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint4, joint4Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint5, joint5Angle, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(clientID, joint6, joint6Angle, sim.simx_opmode_oneshot)
        
        #Position tracking information
        joint1Encoder = sim.simxGetJointPosition(clientID, joint1, sim.simx_opmode_oneshot)
        joint2Encoder = sim.simxGetJointPosition(clientID, joint2, sim.simx_opmode_oneshot)
        joint3Encoder = sim.simxGetJointPosition(clientID, joint3, sim.simx_opmode_oneshot)
        joint4Encoder = sim.simxGetJointPosition(clientID, joint4, sim.simx_opmode_oneshot)
        joint5Encoder = sim.simxGetJointPosition(clientID, joint5, sim.simx_opmode_oneshot)
        joint6Encoder = sim.simxGetJointPosition(clientID, joint6, sim.simx_opmode_oneshot)
        joint1Position = sim.simxGetObjectPosition(clientID, joint1, floor, sim.simx_opmode_oneshot)[1]
        joint2Position = sim.simxGetObjectPosition(clientID, joint2, floor, sim.simx_opmode_oneshot)[1]
        joint3Position = sim.simxGetObjectPosition(clientID, joint3, floor, sim.simx_opmode_oneshot)[1]
        joint4Position = sim.simxGetObjectPosition(clientID, joint4, floor, sim.simx_opmode_oneshot)[1]
        joint5Position = sim.simxGetObjectPosition(clientID, joint5, floor, sim.simx_opmode_oneshot)[1]
        joint6Position = sim.simxGetObjectPosition(clientID, joint6, floor, sim.simx_opmode_oneshot)[1]
        joint1Orientation = sim.simxGetObjectOrientation(clientID, joint1, floor, sim.simx_opmode_oneshot)[1]
        joint2Orientation = sim.simxGetObjectOrientation(clientID, joint2, floor, sim.simx_opmode_oneshot)[1]
        joint3Orientation = sim.simxGetObjectOrientation(clientID, joint3, floor, sim.simx_opmode_oneshot)[1]
        joint4Orientation = sim.simxGetObjectOrientation(clientID, joint4, floor, sim.simx_opmode_oneshot)[1]
        joint5Orientation = sim.simxGetObjectOrientation(clientID, joint5, floor, sim.simx_opmode_oneshot)[1]
        joint6Orientation = sim.simxGetObjectOrientation(clientID, joint6, floor, sim.simx_opmode_oneshot)[1]

        
        '''
        print("J1: Enc: [", round(joint1Encoder[1], 3) ,"] ", \
            "Pos:[", round(joint1Position[0], 3), round(joint1Position[1], 3), round(joint1Position[2], 3), "]", \
            "Rot:[", round(joint1Orientation[0], 3), round(joint1Orientation[1], 3), round(joint1Orientation[2], 3), "]")
        print("J2: Enc: [", round(joint2Encoder[1], 3) ,"] ", \
            "Pos:[", round(joint2Position[0], 3), round(joint2Position[1], 3), round(joint2Position[2], 3), "]", \
            "Rot:[", round(joint2Orientation[0], 3), round(joint2Orientation[1], 3), round(joint2Orientation[2], 3), "]")
        print("J3: Enc: [", round(joint3Encoder[1], 3) ,"] ", \
            "Pos:[", round(joint3Position[0], 3), round(joint3Position[1], 3), round(joint3Position[2], 3), "]", \
            "Rot:[", round(joint3Orientation[0], 3), round(joint3Orientation[1], 3), round(joint3Orientation[2], 3), "]")
        print("J4: Enc: [", round(joint4Encoder[1], 3) ,"] ", \
            "Pos:[", round(joint4Position[0], 3), round(joint4Position[1], 3), round(joint4Position[2], 3), "]", \
            "Rot:[", round(joint4Orientation[0], 3), round(joint4Orientation[1], 3), round(joint4Orientation[2], 3), "]")
        print("J5: Enc: [", round(joint5Encoder[1], 3) ,"] ", \
            "Pos:[", round(joint5Position[0], 3), round(joint5Position[1], 3), round(joint5Position[2], 3), "]", \
            "Rot:[", round(joint5Orientation[0], 3), round(joint5Orientation[1], 3), round(joint5Orientation[2], 3), "]")
        print("J6: Enc: [", round(joint6Encoder[1], 3) ,"] ", \
            "Pos:[", round(joint6Position[0], 3), round(joint6Position[1], 3), round(joint6Position[2], 3), "]", \
            "Rot:[", round(joint6Orientation[0], 3), round(joint6Orientation[1], 3), round(joint6Orientation[2], 3), "]\n")
        
        '''

    #End simulation
    sim.simxFinish(clientID)

else:
    print('Could not connect to remote API server')

#Close all simulation elements
sim.simxFinish(clientID)
cv2.destroyAllWindows()
print('Simulation ended')