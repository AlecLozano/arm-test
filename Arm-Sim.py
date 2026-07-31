# We will do the imports required for this notebook here

# numpy provides import array and linear algebra utilities
import numpy as np

# the robotics toolbox provides robotics specific functionality
import roboticstoolbox as rtb

# spatial math provides objects for representing transformations
import spatialmath as sm

# swift is a lightweight browser-based simulator which comes eith the toolbox
from swift import Swift

# the Python math library
import math

# spatialgeometry is a utility package for dealing with geometric objects
import spatialgeometry as sg

# typing utilities
from typing import Tuple

from ControllerCommon.XboxController import XboxController
import pygame
import time
import os

# Make a panda robot
panda = rtb.models.Panda()

print(panda)


# Make a new environment and add our robot
env = Swift()
env.launch(realtime=True, browser="windows-default")
env.add(panda)

# Change the robot configuration to the ready position
panda.q = panda.qr

# Step the sim to view the robot in this configuration
env.step(0)

# Specify our desired end-effector velocity
ev = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
move_wrist = False
wrist_speed = 0.0

connected = False
while not connected:
    try:
        pygame.init()
        controller = XboxController(deadZone=0.3)
        connected = True
    except pygame.error:
        print("Couldn't connect to controller")
        pygame.joystick.quit()
        time.sleep(1)





# Specify our timestep
dt = 0.05

id = None
val = None
speedCap = .3
wrist_speed_cap = .7

# Run the simulation for 5 seconds
while True:

    inputs = controller.getControllerInput()
    for i in inputs:
        id = controller.getInputID(i)
        val = float(controller.getInputValue(i).strip("\n")) 

    
        if id is not None and val is not None:
            if id == '6':
                ev[0] = val * speedCap
            if id == '5':
                ev[1] = val * speedCap
            if id == '8':
                ev[2] = val*-1 * speedCap
            if id == '9':
                wrist_speed = (1 + val)/2 * wrist_speed_cap
            if id == '10':
                wrist_speed = (1 + val)/2 * wrist_speed_cap * -1
        # if id == '18':
        #     break
        
        # match id:
        #     case '6':
        #         ev[0] = val
        #     case '5':
        #         ev[1] = val
        #     case '8':
        #         ev[2] = val * -1

    # Work out the manipulator Jacobian using the current robot configuration
    J = panda.jacob0(panda.q)

    # Since the Panda has 7 joints, the Jacobian is not square, therefore we must
    # use the pseudoinverse (the pinv method)
    J_pinv = np.linalg.pinv(J)

    Te = panda.fkine(panda.q).A
    
    #Restrict E.E to not drive to the ground
    if Te[2, 3] < 0.1:  # z position below 10cm
        ev[2] = max(ev[2], 0)  # prevent moving further down

    # Calculate the required joint velocities and apply to the robot
    panda.qd = J_pinv @ ev

    #Restrict joint to not bend robot into itself
    panda.q = np.clip(panda.q, panda.qlim[0], panda.qlim[1])


    if wrist_speed != 0:
        panda.qd[5] = wrist_speed   # fixed velocity, not accumulating

    # Step the simulator by dt seconds
    env.step(dt)

    os.system('cls')
    print(wrist_speed)