import scipy
import sympy
import numpy as np
import math
from numpy import linalg as LA
from numpy.linalg import pinv
from math import pi,sin,cos

np.set_printoptions(precision=3,suppress=True)

#Braccio Robot class
class Braccio():
    def __init__(self,l1,l2,l3,d):
        self.DH_Parameters = np.zeros((5,3),dtype=np.float32)
        self.DH_Parameters[0] = np.array([0,-pi/2,l1])
        self.DH_Parameters[1] = np.array([l2,0,0])
        self.DH_Parameters[2] = np.array([l3,0,0])
        self.DH_Parameters[3] = np.array([0,-pi/2,0])
        self.DH_Parameters[4] = np.array([0,0,-d])
     
    #method to compute 4x4 A matrix given q 
    def A_matrix(self,index,q):
        #collect DH parameters
        a,t,d = self.DH_Parameters[index]

        #define A matrix
        A = np.zeros((4,4))
        A[0] = np.array([cos(q), -sin(q)*cos(t), sin(q)*sin(t), a*cos(q)])
        A[1] = np.array([sin(q), cos(q)*cos(t), -cos(q)*sin(t), a*sin(q)])
        A[2] = np.array([0, sin(t), cos(t), d])
        A[3][3] = 1

        return A

    #method to compute forward kinematics
    def FK(self,q_vector):
        #obtain A matrices
        A_matrices = np.zeros((5,4,4))

        for i in range(5):
            A_matrices[i] = self.A_matrix(i,q_vector[i])

        #compute and return end effector transformation matrix
        T = A_matrices[0]

        for i in range(1,5):
            T = np.matmul(T,A_matrices[i])

        return T
    
    #method to compute Geometrical Jacobian
    def Jacobian(self,q_vector):
        #compute A matrices
        A_matrices = np.zeros((5,4,4))

        for i in range(5):
            A_matrices[i] = self.A_matrix(i,q_vector[i])

        #compute T matrices
        T_matrices = np.zeros((5,4,4))
        T_i = A_matrices[0]

        for i in range(0,5):
            T_matrices[i] = T_i
            if (i != 4):
                T_i = np.matmul(T_i,A_matrices[i + 1])

        #compute linear Geometrical Jacobian
        linear_jacobian = np.zeros((3,5))
        EE_pos = T_matrices[4,0:3,3] #extract EE position vector
        
        for i in range(5):
            if i == 0:
                r_vector = EE_pos
                z_vector = np.array([0,0,1])
            else:
                r_vector = EE_pos - T_matrices[i-1,0:3,3]
                z_vector = T_matrices[i-1,0:3,2]

            velocity = np.cross(z_vector,r_vector)                
            linear_jacobian[:,i] = velocity

        return linear_jacobian

    #method to compute inverse kinematics
    def IK(self,target_pos,num_pts,counter_limit):
        error_threshold = 0.02 #define error threshold in m
        q_init = [0,-120,60,115,0] #robot initial configuration
        q_init = np.radians(q_init) #convert from degrees to radians
        T_init = self.FK(q_init)
        init_pos = T_init[0:3,3] 

        #divide target position into trajectory pts
        trajectory_pts = np.zeros((3,num_pts))

        for i in range(num_pts):
            scalar = (i + 1)/num_pts
            trajectory_pts[:,i] = init_pos + scalar*(target_pos - init_pos)

        T_cur = T_init
        q_vector = q_init

        #for loop to iterate through trajectory pts
        for i in range(num_pts):
            counter = 0
            cur_target_pos = trajectory_pts[:,i]

            #while loop for inverse kinematics numerical algorithm
            while(counter < counter_limit):
                #compute position error
                error_vector = cur_target_pos - T_cur[0:3,3]
                error = LA.norm(error_vector)

                #compare with error threshold
                if (error <= error_threshold):
                    break
                
                J = self.Jacobian(q_vector) #compute Geometrical Jacobian
                J_pseudo = pinv(J) #compute pseudo inverse
                qdot = np.matmul(J_pseudo,error_vector.reshape(3,1)) #compute qdot
                qdot = qdot.flatten()
                
                #update q_vector
                if (error <= 0.05):
                    dt = 1
                    q_vector = q_vector + dt*qdot
                else:
                    dt = 0.5
                    q_vector = q_vector + dt*qdot

                #compute forward kinematics of new q_vector
                T_cur = self.FK(q_vector)

                counter = counter + 1 #increment counter

        return T_cur, q_vector, error
        
robot = Braccio(0.0715,0.125,0.125,0.192)
target_pos = [0.35,-0.25,0.1]
counter_limit = 70
num_pts = 15

T,q,error = robot.IK(target_pos,num_pts,counter_limit)

print(T)
print("q vector")
print(np.rad2deg(q))
print("error")
print(error)
