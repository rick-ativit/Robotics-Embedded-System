Loop = True
c = " " # Input condition for asking again
n = 1 # No. of person input
Total_weight = 0
while n!=7:
    Weight = input ("Enter weight of Person "+str(n)+c+"in kg:")
    Weight = int(Weight)
    Total_weight = Total_weight + Weight
    print("Total weight of "+str(n)+" persons = "+str(Total_weight))
    if Total_weight>=450:
        print("WARNING!:"+" Total weight is more than 450kg")
        Total_weight = Total_weight - Weight
        c = " again "
    else:
        n = n+1
        c = " "