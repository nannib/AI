import random
import csv
i=0

while i<3000:
      i=i+1
      age=random.randint(10, 60)
      age2=random.randint(10, 60)
      breed=random.randint(0,1)
      #if b==2:
        #  b=1
      crimes=random.randint(0,10)
      crimes2=random.randint(0,10)
      rand_badterrestrial=random.randint(1,10)
      if (rand_badterrestrial==2 or rand_badterrestrial==5 or rand_badterrestrial==7 or rand_badterrestrial==9) or (crimes>crimes2 and breed==2) or (age<age2 and breed==2):
          label="yes"
      else:
          label="no"
      print(age,",",breed,",",crimes,",",label)
        

      header = ['age', 'breed', 'past_crimes', label]
      data = [age,breed,crimes,label]

      with open('disct01.csv', 'a', encoding='UTF8', newline='') as f:
          writer = csv.writer(f)

        # write the header
          #writer.writerow(header)

        # write the data
          writer.writerow(data)
random.seed()

