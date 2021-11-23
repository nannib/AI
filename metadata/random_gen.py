import random
import csv

header = ['file_type', 'path_depth', 'size', 'namefile_lenght','occurences']
with open('metadata.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
i=0
s=0
while i<3000:
      i=i+1
      type_f=random.randint(1, 6)
      depth=random.randint(1,10)
      size=random.randint(0,10000)
      lenght_nf= random.randint(1,20)
      occurences=random.randint(0,200)
      if ((type_f==2) or (type_f==3) or (type_f==4)) and (depth>3) and (size>1024 and size<8048) and (lenght_nf>2 and lenght_nf<15) and (occurences>10):
          label="suspect"
          s=s+1
      else:
          label="clean"
      #print(type_f,",",depth,",",size,",",lenght_nf,",",occurences,",",label)
      #print (s)
      data=[type_f,depth,size,lenght_nf,occurences,label]
      with open('metadata.csv', 'a', encoding='UTF8', newline='') as f:
          writer = csv.writer(f)
          writer.writerow(data)
random.seed()

