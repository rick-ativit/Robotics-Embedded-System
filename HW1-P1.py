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

#Step 2
f=open('CurrentDateTime.txt','w')
f.write(var1) 
f.write(var2) 
f.close()

#Step 3
f=open('CurrentDateTime.txt')
s1=f.readline(26) #Read
s2=f.readline()
f.close()

#Step 4
#Convert str -> datetime
s1d = datetime.strptime(s1,"%Y-%m-%d %H:%M:%S:%f")
s2d = datetime.strptime(s2,"%Y-%m-%d %H:%M:%S:%f") 
print(s1d) #Print current date
print(s2d) #Print random date
DeltaT = s1d - s2d
print(DeltaT) #Print time difference

