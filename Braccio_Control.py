import numpy as np
from numpy import linalg as LA
from numpy.linalg import pinv
from math import pi,sin,cos,atan2,sqrt,radians,degrees

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

    #method to compute inverse kinematics numerically
    def IK(self,target_pos,num_pts,counter_limit,q_init = [0,-120,60,115,0]):
        error_threshold = 0.01 #define error threshold in m
        #q_init = [0,-120,60,115,0] #robot initial configuration
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

    def limit(self,value,min,max):
        if (value < min):
            return min
        elif (value > max):
            return max
        else:
            return value

    #geometrical inverse kinematics with joint limit constraints
    def geo_IK(self,x,y,z):
        success = False #flag variable for IK success
        #define link lengths (m)
        l1 = 0.0715
        l2 = 0.125
        l3 = 0.125
        l4 = 0.192
        #x-y plane
        q1rad = atan2(y,x)
        q1 = degrees(q1rad)
        r = x*x + y*y 
        r = sqrt(r)
        s = z - l1
        a_y = -1
        b_y = -1
        EE_y = -1
        #arm plane
        for phi in range(-90,1): #iterate through valid phi in degrees
            phi_rad = radians(phi) #convert phi from degrees to radians
            x_prime = r - l4*cos(phi_rad)
            y_prime = s - l4*sin(phi_rad)
            d = sqrt(x_prime*x_prime + y_prime*y_prime)
            #check if (x',y') is within workspace limits
            if (d > (l2 + l3) or (d < abs(l2 - l3))):
                continue
            #if (x',y') is reachable -> solve for q2,q3,q4 for elbow down config
            #q3
            D = ((d**2) - (l2**2) - (l3**2))/(2*l2*l3)
            q3rad = atan2(-sqrt(1 - (D**2)),D)
            q3 = degrees(q3rad)
            if (q3 != self.limit(q3,-90,90)):
                continue
            #q2
            a = l2 + l3*cos(q3rad)
            b = l3*sin(q3rad)
            term1 = atan2(y_prime,x_prime)
            term2 = atan2(b,a)
            q2rad = term1 - term2
            q2 = degrees(q2rad)
            if (q2 != self.limit(q2,15,165)):
                continue
            #q4
            q4 = phi - q2 - q3
            if (q4 != self.limit(q4,-90,90)):
                continue
            #check if robot arm crashes into ground
            a_y = l1 + l2*sin(q2rad) #pt a y position
            if ((a_y < 0) or (a_y == 0)):
                continue
            b_y = a_y + l3*sin(q2rad + q3rad) #pt b y position
            if ((b_y < 0) or (b_y == 0)):
                continue
            q4rad = radians(q4)
            EE_y = b_y + l4*sin(q2rad + q3rad + q4rad) #EE y position
            if (EE_y < 0):
                continue
            #remap angles to physical servo angles + update flag variable
            success = True
            q1 = -q1 + 90
            q2 = -q2 + 180
            q3 = -q3 + 90
            q4 = -q4 + 90
            break
        
        q = [90,90,90,90,90] #home config if IK fails
        if (success == True):
            q = [q1,q2,q3,q4,90]

        return success,q,phi,a_y,b_y,EE_y

    #method for q remapping to servo angles for numerical IK
    def q_remapping(self,q_vector):
        q_vector[0] = -q_vector[0] + 90 #M1
        q_vector[1] = q_vector[1] + 180 #M2
        q_vector[2] = q_vector[2] + 90 #M3
        q_vector[4] = q_vector[4] + 90 #M5

        return q_vector
