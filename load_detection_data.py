import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# generate data.txt for detection time
# plot from data.txt
######################################
step = 0
for ii in range(30):
    detection_time_of_brand = []
    if step >= 20:
        print(f'@@ Now generate data.txt')
        break
    step += 1


headers = ['brand','duration','result']
df = pd.read_csv('data/detection_data.csv',names=headers)

print(df)
a = open('data/detection_data.csv')

b = a.readlines()
print(f'readlines : {b} len(b):{len(b)}')

brand1 = []
brand2 = []
brand3 = []
brand4 = []
brand5 = []

for i in range(1, len(b)):
    temp = b[i].split(',')
    print(f'temp: {temp}')
    brand1.append(temp[0])
    brand2.append(temp[1])


brand1 = list(map(float,brand1))
brand2 = list(map(float,brand2))
Device = ['Apple','XBOX','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure()
plt.plot(np.arange(6),brand1,label='Apple')
plt.plot(np.arange(6),brand2,label='XBOX')

plt.legend()
plt.xlabel('count',size=18)
plt.ylabel('duration $^\circ$C',size=18)

plt.savefig('test.png',dpi=300)

plt.show()