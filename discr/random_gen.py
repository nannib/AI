import random
i=0

while i<300:
      i=i+1
      age=random.randint(10, 60)
      breed=random.randint(0,1)
      #if b==2:
        #  b=1
      crimes=random.randint(0,10)
      if (crimes>5) or (crimes>2 and breed==1) or (age<21 and breed==1):
          label="yes"
      else:
          label="no"
      print(age,",",breed,",",crimes,",",label)
     
random.seed()

