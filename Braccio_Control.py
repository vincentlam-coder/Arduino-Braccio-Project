from math import pi
from roboticstoolbox import DHRobot, RevoluteDH

#define robot dimensions (m)
l1 = 0.0715
l2 = 0.125
l3 = 0.125
d = 0.192

# Define the links using standard DH parameters (d, alpha, a, offset)
# The offset parameter (theta) is the joint variable for a revolute joint
links = [
    RevoluteDH(a=0, alpha=-pi/2, d=l1, offset=0, qlim=[-90 * (pi/180), 90 * (pi/180)]),
    RevoluteDH(a=l2, alpha=0, d=0, offset=-pi/2, qlim=[-180 * (pi/180), 0 * (pi/180)]),
    RevoluteDH(a=l3, alpha=0, d=0, offset=0, qlim=[-90 * (pi/180), 90 * (pi/180)]),
    RevoluteDH(a=0, alpha=-pi/2, d=0, offset=pi/2, qlim=[0 * (pi/180), 180 * (pi/180)]),
    RevoluteDH(a=0, alpha=0, d=-d, offset=0, qlim=[-90 * (pi/180), 90 * (pi/180)]),
]

# Assemble the links into a DHRobot model
robot = DHRobot(links, name='Braccio')

# Print the robot details
print(robot)

# Calculate forward kinematics for zero joint angles
q = [0,-pi/2,0,pi/2,0]
T = robot.fkine(q)
print("Forward kinematics T matrix:\n", T)
