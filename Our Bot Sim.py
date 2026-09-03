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


#ETS Representation of our robot
E1 = rtb.ET.Rz() #J1 rotates about z-axis

E2 = rtb.ET.tz(0.2032)#J2-J3 length (m)
E3 = rtb.ET.Ry()#J2 rotates around y-axis

E4 = rtb.ET.tz(0.2286)#J3-J4 (m)
E5 = rtb.ET.Ry()# J3 spins about y-axis

E6 = rtb.ET.tz(0.0920242)#J4-J5 (m)
E7 = rtb.ET.Ry()#J4 spins about y-axis

ArrrmBot = rtb.Robot(E1 * E2 * E3 * E4 * E5 * E6 * E7) 

print(ArrrmBot)

# Change the robot configuration to a reasonable starting pose
ArrrmBot.q = [0.0, 0.3, -0.6, -0.3]
print(ArrrmBot.q)

# ArrrmBot is a bare ETS chain - it has no URDF/mesh model, so Swift has
# nothing to draw by default. Build a simple "stick figure": one cylinder
# per rigid link segment plus a sphere at each joint, and reposition them
# by hand every step using forward kinematics. These are added as plain
# shapes (not attached to link.geometry) so we fully control their pose.
link_lengths = [0.2032, 0.2286, 0.0920242]  # matches E2, E4, E6 above
link_radius = 0.02
joint_radius = 0.035

link_shapes = [
    sg.Cylinder(radius=link_radius, length=length, color=(0.2, 0.4, 0.8, 1.0))
    for length in link_lengths
]
joint_shapes = [
    sg.Sphere(radius=joint_radius, color=(0.8, 0.2, 0.2, 1.0))
    for _ in range(ArrrmBot.n)
]


def _align_z_to(direction: np.ndarray) -> sm.SE3:
    """Rotation that points the +z axis (a cylinder's long axis) along `direction`."""
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return sm.SE3()

    z_axis = np.array([0.0, 0.0, 1.0])
    d = direction / length
    dot = np.clip(np.dot(z_axis, d), -1.0, 1.0)
    axis = np.cross(z_axis, d)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-9:
        # direction is already +z or exactly -z
        return sm.SE3() if dot > 0 else sm.SE3.Rx(np.pi)

    return sm.SE3.AngleAxis(np.arccos(dot), axis / axis_norm)


def update_robot_shapes(robot: rtb.Robot):
    """Move the cylinders/spheres to match the robot's current joint origins."""
    # fkine_all returns one SE3 per frame, starting with the world/base frame
    # (index 0) followed by one per link output. Since J1 (Rz) doesn't
    # translate, the base and link0's output are the same point, so we drop
    # index 0 and keep the 4 link-output points: [J1, J2, J3, EE].
    joint_origins = [T.t for T in robot.fkine_all(robot.q)][1:]

    for i, sphere in enumerate(joint_shapes):
        sphere.T = sm.SE3(joint_origins[i]).A

    for i, rod in enumerate(link_shapes):
        p0, p1 = joint_origins[i], joint_origins[i + 1]
        mid = (p0 + p1) / 2
        rod.T = (sm.SE3(mid) * _align_z_to(p1 - p0)).A


# pyplot wasn't redrawing live, so drive the sim through Swift instead - same
# ETS-built robot, just a different (browser-based) viewer/backend.
env = Swift()
env.launch(realtime=True, browser="windows-default")
env.add(ArrrmBot)
for shape in link_shapes + joint_shapes:
    env.add(shape)

update_robot_shapes(ArrrmBot)
env.step(0)

# Specify our desired end-effector velocity
ev = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
wrist_speed = 0.0
J1_speed = 0.0


#Wait for xbox controller connection
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

# ArrrmBot only has 4 joints (Rz, Ry, Ry, Ry) - there's no separate wrist
# joint like the Panda's, so the last joint (index -1) doubles as "wrist".
wrist_index = ArrrmBot.n - 1

while True:

    inputs = controller.getControllerInput()
    for i in inputs:
        id = controller.getInputID(i)
        val = float(controller.getInputValue(i).strip("\n"))

        if id is not None and val is not None:
            #Move along x-axis
            if id == '6':
                ev[0] = val * speedCap
            #Move along y-axis
            if id == '5':
                ev[1] = val * speedCap
            #Rotate along z-axis
            if id == '8':
                J1_speed = val * -1
            #Move Wrist forward
            if id == '9':
                wrist_speed = (1 + val) / 2 * wrist_speed_cap
            #Move Wrist backward
            if id == '10':
                wrist_speed = (1 + val) / 2 * wrist_speed_cap * -1

    # Work out the manipulator Jacobian using the current robot configuration
    J = ArrrmBot.jacob0(ArrrmBot.q)

    # ArrrmBot has 4 joints so the Jacobian is not square, use the
    # pseudoinverse (the pinv method) same as the Panda control scheme
    J_pinv = np.linalg.pinv(J)

    Te = ArrrmBot.fkine(ArrrmBot.q).A

    #Restrict E.E to not drive to the ground
    if Te[2, 3] < 0.1:  # z position below 10cm
        ev[2] = max(ev[2], 0)  # prevent moving further down

    print(J_pinv)
    print(ev)
    # Calculate the required joint velocities and apply to the robot
    ArrrmBot.qd = J_pinv @ ev

    #Restrict joint to not bend robot into itself
    ArrrmBot.q = np.clip(ArrrmBot.q, ArrrmBot.qlim[0], ArrrmBot.qlim[1])

    if wrist_speed != 0:
        ArrrmBot.qd[wrist_index] = wrist_speed   # fixed velocity, not accumulating

    if J1_speed != 0:
        ArrrmBot.qd[0] = J1_speed 
    # Step the simulator by dt seconds, then move our hand-drawn "model" to
    # match the robot's new joint configuration
    env.step(dt)
    update_robot_shapes(ArrrmBot)