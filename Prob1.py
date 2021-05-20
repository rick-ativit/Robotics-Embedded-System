import numpy as np                   # import NumPy module
from matplotlib import pyplot as plt # import matplotlib

class KalmanFilter(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        dt: sampling time (time for 1 cycle)
        u_x, u_y: acceleration in x-direction and y-direction
        std_acc: standard deviation of acceleration or process noise magnitude (Unit: m/s) 
        x_std_meas, y_std_meas: standard deviation of the measurement in x-direction and y-direction
        """
        self.dt = dt # discrete time step (Unit: sec)
        self.u = np.matrix([[u_x],[u_y]]) # discrete control input (Unit: m/s^2)

        # Define the State Transition Matrix : A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0,       self.dt],
                            [0, 0, 1,       0],
                            [0, 0, 0,       1]])
        # Define the Control Input Matrix : B
        self.B = np.matrix([[0, 0             ],
                            [0,             -(self.dt**2)/2],
                            [0,        0             ],
                            [0,              -self.dt       ]])
        
        # Define Transformation matrix : H 
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        # Initialize Process Noise Covariance Matrix : Q
        self.Q = np.matrix([[(self.dt**4)/4, 0,              (self.dt**3)/2, 0],
                            [0,              (self.dt**4)/4, 0,              (self.dt**3)/2],
                            [(self.dt**3)/2, 0,              self.dt**2,     0],
                            [0,              (self.dt**3)/2, 0,              self.dt**2]]) * std_acc**2
        # Initialize Measurement Noise Covariance Matrix : R
        self.R = np.matrix([[x_std_meas**2, 0            ],
                           [0,              y_std_meas**2]])
        self.P = np.eye(self.A.shape[1]) # Initialize Error Covariance: P
        self.x = np.matrix([[0], [1], [70.71], [70.71]]) # # Intial State Vector: X

    def PredictUpdate(self):
        # Prediction Update for state at each time step
        # x_k =Ax_(k-1) + Bu_(k-1) : Eq.(1)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate Priori Error Covariance:
        # P= A*P*A' + Q : Eq.(2)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def MeasurementUpdate(self, z):
        # Measurement Update for compute Kalman Gain
        # S = H*P*H'+R : Eq.(3) 
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R) : Eq.(3) 
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  
        # Eq.(4): Calculate Posteriori State Estimate
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))  
        # Eq.(5): Calculate Posteriori Error Covariance 
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P  
        return self.P
    
dt  = 0.1 # discrete time step (Unit: sec)
u_x = 0 # constant acceleration (Unit: m/s^2)
u_y = 9.81
t = np.arange(0, 10, dt) # discrete time
Ideal_Motion_x = (70.71*t) # generate an Ideal Motion Path
Ideal_Vel_x = 70.71+(u_x*t) # generate an Ideal Velocity
Ideal_Motion_y = (70.71*t)-(0.5*u_y*(t**2)) # generate an Ideal Motion Path
Ideal_Vel_y = (70.71**2+2*(-u_y)*Ideal_Motion_y)**0.5 # generate an Ideal Velocity


std_acc = 0.2    # the standard deviation of the acceleration (Unit: m/s^2)
x_std_meas = 0.3    # the standard deviation of the position measurement noise (Unit: m)
y_std_meas = 0.3
# create KalmanFilter object
kf = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

predictions_x = []  # initialize empty prediction update
predictPos_x = []   # initialize Position prediction
predictVel_x = []   # initialize Velocity prediction
measurements_x= [] # initialize empty measurement update
PosMeasurements_x =[] # initialize Position measurement
ErrorCovPos_x= []  # intitialize empty priori error covariance of Position
ErrorCovVel_x= []  # intitialize empty priori error covariance of Velocity
i=0

for x in Ideal_Motion_x:
    # Mesurement at each time step
    z = (kf.H * x) + np.random.normal(0, 50)
    PosMeasurements_x.append(z.item(0))
    
    # Kalman Filter: Prediction Update
    predictions_x.append(kf.PredictUpdate())
    
    
    predictPos_x.append(predictions_x[i].item((0)))
    predictVel_x.append(predictions_x[i].item((2)))
   
    # Kalman Filter: Measurement Update
    measurements_x.append(kf.MeasurementUpdate(z.item(0)))
    ErrorCovPos_x.append(measurements_x[i].item(0))
    ErrorCovVel_x.append(measurements_x[i].item(2))
    i=i+1

MeasurementError_x = PosMeasurements_x - np.array(Ideal_Motion_x) 
KFPosPredictError_x = predictPos_x - np.array(Ideal_Motion_x)
KFVelPredictError_x = predictVel_x - np.array(Ideal_Vel_x)

plt.figure(1)
plt.title('Kalman filter for tracking a moving object in 1-D motion', fontsize=14)
plt.plot(t, PosMeasurements_x, label='Measurements', color='b',linewidth=0.5)
plt.plot(t, np.array(Ideal_Motion_x), label='Ideal Position', color='g', linewidth=1.5)
plt.plot(t, predictPos_x, label='Kalman Filter Prediction', color='r', linewidth=1.5)
plt.xlabel('Time (sec)', fontsize=14), plt.ylabel('Position (m)', fontsize=14)
plt.legend()

plt.figure(2)
plt.title('Kalman filter for tracking a moving object in 1-D motion', fontsize=14)
plt.plot(t, np.array(Ideal_Vel_x), label='Ideal Velocity', color='g', linewidth=1.5)
plt.plot(t, predictVel_x, label='Kalman Filter Prediction', color='r', linewidth=1.5)
plt.xlabel('Time (sec)', fontsize=14), plt.ylabel('Velocity (m/s)', fontsize=14)
plt.legend()

# Next loop
kf = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

predictions_y = []  # initialize empty prediction update
predictPos_y = []   # initialize Position prediction
predictVel_y = []   # initialize Velocity prediction
measurements_y= [] # initialize empty measurement update
PosMeasurements_y =[] # initialize Position measurement
ErrorCovPos_y= []  # intitialize empty priori error covariance of Position
ErrorCovVel_y= []  # intitialize empty priori error covariance of Velocity
i=0

for y in Ideal_Motion_y:
    # Mesurement at each time step
    z = (kf.H * y) + np.random.normal(0, 50)
    PosMeasurements_y.append(z.item(0))
    
    # Kalman Filter: Prediction Update
    predictions_y.append(kf.PredictUpdate())
    
    
    predictPos_y.append(predictions_y[i].item((1)))
    predictVel_y.append(predictions_y[i].item((3)))
   
    # Kalman Filter: Measurement Update
    measurements_y.append(kf.MeasurementUpdate(z.item(0)))
    ErrorCovPos_y.append(measurements_y[i].item(1))
    ErrorCovVel_y.append(measurements_y[i].item(3))
    i=i+1

MeasurementError_y = PosMeasurements_y - np.array(Ideal_Motion_y) 
KFPosPredictError_y= predictPos_y - np.array(Ideal_Motion_y)
KFVelPredictError_y = predictVel_y - np.array(Ideal_Vel_y)

plt.figure(3)
plt.title('Kalman filter for tracking a moving object in 1-D motion', fontsize=14)
plt.plot(t, PosMeasurements_y, label='Measurements', color='b',linewidth=0.5)
plt.plot(t, np.array(Ideal_Motion_y), label='Ideal Position', color='g', linewidth=1.5)
plt.plot(t, predictPos_y, label='Kalman Filter Prediction', color='r', linewidth=1.5)
plt.xlabel('Time (sec)', fontsize=14), plt.ylabel('Position (m)', fontsize=14)
plt.legend()

plt.figure(4)
plt.title('Kalman filter for tracking a moving object in 1-D motion', fontsize=14)
plt.plot(t, np.array(Ideal_Vel_y), label='Ideal Velocity', color='g', linewidth=1.5)
plt.plot(t, predictVel_y, label='Kalman Filter Prediction', color='r', linewidth=1.5)
plt.xlabel('Time (sec)', fontsize=14), plt.ylabel('Velocity (m/s)', fontsize=14)
plt.legend()

plt.show()


