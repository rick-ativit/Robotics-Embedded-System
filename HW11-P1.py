import tensorflow as tf         # import tensorflow module
import matplotlib.pyplot as plt # import matplotlib module
import numpy as np              # import numpy module 
# Define Goldstein-Price Function
def Gold(x1, x2): 
    return (1+((x1+x2+1)**2)*(19-(14*x1)+(3*(x1**2))-(14*x2)+(6*x1*x2)+(3*(x2**2)))) * (30+(((2*x1)-(3*x2))**2)*(18-(32*x1)+(12*(x1**2))+(48*x2)-(36*x1*x2)+(27*(x2**2))))
def Gold_minimzie():
    return (1+((x1+x2+1)**2)*(19-(14*x1)+(3*(x1**2))-(14*x2)+(6*x1*x2)+(3*(x2**2)))) * (30+(((2*x1)-(3*x2))**2)*(18-(32*x1)+(12*(x1**2))+(48*x2)-(36*x1*x2)+(27*(x2**2))))
# Initialize function to reset model parameters
def reset():
    x1 = tf.Variable(0.0)  
    x2 = tf.Variable(1.0)  
    return x1, x2

x1, x2 = reset()
opt = tf.keras.optimizers.SGD(learning_rate=0.0000045, momentum=0.9)
# Setup variable for contour plot
x1_axis = np.arange(-1.5, 1.5, 0.01) 
x2_axis = np.arange(-1.5, 1.5, 0.01) 
[X1, X2] = np.meshgrid(x1_axis, x2_axis)
Y = (1+((X1+X2+1)**2)*(19-(14*X1)+(3*(X1**2))-(14*X2)+(6*X1*X2)+(3*(X2**2)))) * (30+(((2*X1)-(3*X2))**2)*(18-(32*X1)+(12*(X1**2))+(48*X2)-(36*X1*X2)+(27*(X2**2))))
# Plot contour of function
plt.figure(1)
levels = [1.0,5.0,10.0,20.0,40.0,80.0,120.0,500.0,1000.0,4000.0,15000.0,40000.0,9e4,20e4]
contour = plt.contour(X1, X2, Y, levels) 
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
plt.title(r"Contour Plot of Goldstein-Price Function") 
plt.xlabel(r"$x_1$"), plt.ylabel(r"$x_2$") 
 
Z = []; Niter = []
for i in range(1000):
    print ('y = {:.1f}, x1 = {:.1f}, x2 = {:.1f}'.format(Gold(x1, x2).numpy(), x1.numpy(), x2.numpy()))
    opt.minimize(Gold_minimzie, var_list=[x1, x2])
    Z.append(Gold(x1,x2)); Niter.append(i)
    if (i % 20 == 0):
        plt.plot(x1,x2, 'ro', markersize=4)
#         plt.pause(0.1)
plt.show()

plt.figure(2)
plt.plot(Niter,Z,'b-')
plt.title('Convergence rate of Adaptive Moments (SGD) Optimizer')
plt.xlabel('Iteration'), plt.ylabel('Y value')
plt.show()