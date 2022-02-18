import random

i=0

header = "age,"+"breed,"+"past_crimes,"+"suspect"
print(header)
db = open("db.csv", "w")
print(header,file=db)

while i<200:
      i=i+1
      age=random.randint(10, 60)
      age2=random.randint(10, 60)
      breed=random.randint(0,1)
      if breed==0:
          strbreed="Martian"
      else:
          strbreed="Terrestrial"
      crimes=random.randint(0,10)
      crimes2=random.randint(0,10)
      rand_badterrestrial=random.randint(1,10)
      if (rand_badterrestrial==2 or rand_badterrestrial==5 or rand_badterrestrial==7 or rand_badterrestrial==9) or (crimes>crimes2 and breed==0) or (age<age2 and breed==0):
          label="yes"
      else:
          label="no"
      print(age,",",strbreed,",",crimes,",",label)
        

      
      data = str(age)+","+strbreed+","+str(crimes)+","+label

      print(data,file=db)

db.close()
random.seed()

