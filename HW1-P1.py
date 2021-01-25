#Step 1
from datetime import datetime
import random as R
d = datetime.now()
#randomize year from 2000-now
start = datetime(2000, 1, 1, 00, 00, 00)
end = d
d_random = start + (end - start) * R.random()
#Convert datetime -> str
var1 = d.strftime("%Y-%m-%d %H:%M:%S:%f")
var2 = d_random.strftime("%Y-%m-%d %H:%M:%S:%f")
var2i= R.randint(0,99999)

#Step 2
f=open('CurrentDateTime.txt','w')
f.write(var1) 
f.write(var2)
f.write(str(var2i))
f.close()

#Step 3
f=open('CurrentDateTime.txt')
s1=f.readline(26) #Current datetime
s2=f.readline(26) #Random datetime
s2i=f.readline() #Random integer
f.close()

#Step 4
#Convert str -> datetime
s1d = datetime.strptime(s1,"%Y-%m-%d %H:%M:%S:%f") #Current datetime format
s1i = int(s1[-6:]) #Current, milliseconds, int
s2d = datetime.strptime(s2,"%Y-%m-%d %H:%M:%S:%f") #Random datetime format
s2i = int(s2i) #Random, milliseconds, int
print("Current datetime: "+str(s1d)) #Print current date
print("Random datetime: "+str(s2d)) #Print random date
print("Random integer: "+str(s2i)) #Print random int
DeltaT = s1d - s2d
DeltaT2 = s1i - s2i
print("DeltaT from current & random datetime: ",DeltaT) #Print time difference
print("DeltaT from current time & int: 0.",DeltaT2)

