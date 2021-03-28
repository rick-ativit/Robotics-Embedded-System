from matplotlib import pyplot as plt # import matplotlib
import numpy as np # import NumPy module
import math as m

# Function for Measurement Update
def MeasureUpdate(mu1, var1, mu2, var2):
    ''' This function takes 2 means and 2 variance parameters
        and returns measurement update parameters.'''
    new_mu = (mu1*var2 + mu2*var1) / (var1 + var2)
    new_var = 1 / ((1 / var1) + (1 / var2))
    return [new_mu, new_var]
   
# Function for Prediction/Motion Update   
def PredictUpdate(mu1, var1, mu2, var2):
    ''' This function takes 2 means and 2 variance parameters
        and returns prediction update parameters.'''
    new_mu = mu1 + mu2
    new_var = var1 + var2
    return [new_mu, new_var]

# Given Information
t = np.arange(5,40,1)
IdealMotion = 0.1*((t**2)-t)
measurement_var = 5.2**2 # measurement (r^2)
motion_var = 0.5**2 # math-model motion (sigma^2)
# Step 1
measurement_mu = IdealMotion + np.random.normal(0,m.sqrt(measurement_var),len(t)) # 35 entries
motion_mu = np.diff(IdealMotion) + np.random.normal(0,m.sqrt(motion_var),len(t)-1) # 34 entries

# Step 2
mu,var = 4.1, 8.5**2 # initial mean and variance
mu_list=[]
var_list=[]
Predict = True
x = np.arange(-10,160,0.1)
for n in range(len(t)):
    # measurement update with measured GPS-sensor uncertainty
    if n == (len(t)-1):
        Predict = False
    mu, var = MeasureUpdate(mu, var, measurement_mu[n], measurement_var)
    print('Measurement Update: [{}, {}]'.format(mu,var))
    if Predict == True:
        mu, var = PredictUpdate(mu, var, motion_mu[n], motion_var)
        print('Prediction Update: [{}, {}]'.format(mu,var))
    mu_list.append(mu)
    var_list.append(var)
    ProbDist = np.exp(-np.square(x-mu)/(2*var))/(np.sqrt(2*np.pi*var))
    plt.plot(x,ProbDist,'k-')
    plt.ylabel('Probability Distribution'), plt.xlabel('X')
plt.show()
mu_list = np.array(mu_list)
var_list = np.array(var_list)
mu_ref = np.copy(mu_list)
var_ref = np.copy(var_list)

plt.subplot(211)
plt.plot(t,IdealMotion,'b--')
plt.plot(t,mu_list,'r-')
plt.ylabel('position [m]')
plt.xlabel('time (s)')
plt.legend(('Ideal Motion','Mean Position'))
plt.subplot(212)
plt.plot(t[:-1],var_list[:-1])
plt.ylabel('Variance [m^2]')
plt.xlabel('time (s)')
plt.show()
print("End of Step 2\t")

# Step 3
measurement_var = 3.2 # measurement (r)
measurement_mu = IdealMotion + np.random.normal(0,measurement_var,len(t))
measurement_mu[18:22]=0
mu,var = 4.1, 8.5**2 # initial mean and variance
mu_list=[]
var_list=[]
Predict = True
x = np.arange(-5,160,0.1)
for n in range(len(t)):
    # measurement update with measured GPS-sensor uncertainty
    if n == (len(t)-1):
        Predict = False
    mu, var = MeasureUpdate(mu, var, measurement_mu[n], measurement_var**2)
    print('Measurement Update: [{}, {}]'.format(mu,var))
    if Predict == True:
        mu, var = PredictUpdate(mu, var, motion_mu[n], motion_var)
        print('Prediction Update: [{}, {}]'.format(mu,var))
    mu_list.append(mu)
    var_list.append(var)
    ProbDist = np.exp(-np.square(x-mu)/(2*var))/(np.sqrt(2*np.pi*var))
    plt.plot(x,ProbDist,'k-')
    plt.ylabel('Probability Distribution'), plt.xlabel('X')
plt.show()
mu_list = np.array(mu_list)
var_list = np.array(var_list)

plt.subplot(211)
plt.plot(t,IdealMotion,'b--')
plt.plot(t,mu_list,'r-')
plt.ylabel('position [m]')
plt.xlabel('time (s)')
plt.legend(('Ideal Motion','Mean Position'))
plt.subplot(212)
plt.plot(t[:-1],var_list[:-1])
plt.ylabel('Variance [m^2]')
plt.xlabel('time (s)')
plt.show()

# Step 4
plt.subplot(221)
plt.plot(t,IdealMotion,'b--')
plt.plot(t,mu_ref,'r-')
plt.ylabel('position [m]')
plt.xlabel('time (s)')
plt.legend(('Ideal Motion','Mean Position'))
plt.subplot(223)
plt.plot(t[:-1],var_ref[:-1])
plt.ylim(0, 25)
plt.ylabel('Variance [m^2]')
plt.xlabel('time (s)')
plt.subplot(222)
plt.plot(t,IdealMotion,'b--')
plt.plot(t,mu_list,'r-')
plt.ylabel('position [m]')
plt.xlabel('time (s)')
plt.legend(('Ideal Motion','Mean Position'))
plt.subplot(224)
plt.plot(t[:-1],var_list[:-1])
plt.ylim(0, 25)
plt.ylabel('Variance [m^2]')
plt.xlabel('time (s)')
plt.show()

