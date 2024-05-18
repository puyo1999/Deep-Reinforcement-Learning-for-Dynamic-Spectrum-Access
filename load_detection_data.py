import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import random

headers = ['brand', 'detection']
brand_name = ["Apple", "XBOX", "Roku", "Amazon", "Samsung", "Sony", "Yamaha", "Oppo", "Olleh", "Btv", "U+tv", "Dish", "DirecTV"]
codeset_name = ["", "", "", ""]
detection_time = []

# generate random data for detection time
######################################

data = []
data_dict = {}

for ii in range(100):
    brand = random.choice(brand_name)
    detection = random.randint(6, 30)
    data.append([brand, detection])

    if brand not in data_dict:
        data_dict[brand] = detection
    else:
        data_dict[brand] = (data_dict[brand] + detection)/2

    #data_dict[brand] = detection


data_dict = dict(sorted(data_dict.items(), key=lambda item: item[1]))

print(f" data : {data}")
print(f" data_dict : {data_dict}")

with open("data/random_detection_data.csv", "w", newline="") as csvfile:
    mbr_writer = csv.writer(csvfile, delimiter=',')
    mbr_writer.writerow(['brand', 'detection'])
    mbr_writer.writerows(data)

with open("data/random_detection_data_dict.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames= headers)
    writer.writeheader()
    for brand, detection in data_dict.items():
        writer.writerow({'brand': brand, 'detection': detection})


#headers = ['brand','duration','result']
#df = pd.read_csv('data/detection_data.csv',names=headers)
#df = pd.read_csv('data/detection_data.csv')
df = pd.read_csv('data/brand_codeset_detection_data.csv')
print(df)

df_random = pd.read_csv('data/random_detection_data.csv')
df_random.head()
print(f'df_random.head() :\n{df_random.head()}')

df_mean = df_random.groupby('brand')['detection'].agg(**{'mean_detection':'mean'}).reset_index()
print(f'after groupby with mean() :\n{df_mean}')


a = open('data/brand_codeset_detection_data.csv')

b = a.readlines()
print(f'readlines : {b} len(b):{len(b)}')

codeset = []
name = []
brand1 = []
brand2 = []
brand3 = []
brand4 = []
brand5 = []
brand6 = []
brand7 = []
brand8 = []

for i in range(1, len(b)):
    temp = b[i].split(',')
    print(f'temp: {temp}')

    if i == 1:
        codeset.append(temp)
    elif i == 2:
        name.append(temp)
    else:
        brand1.append(temp[0])
        brand2.append(temp[1])
        brand3.append(temp[2])
        brand4.append(temp[3])
        brand5.append(temp[4])
        brand6.append(temp[5])
        brand7.append(temp[6])
        brand8.append(temp[7])

print(f'codeset: {codeset}')
print(f'name: {name}')
print(f'brand1: {brand1}')
print(f'brand2: {brand2}')
print(f'brand3: {brand3}')
print(f'brand4: {brand4}')
print(f'brand5: {brand5}')
print(f'brand6: {brand6}')
print(f'brand7: {brand7}')
print(f'brand8: {brand8}')
brand1 = list(map(float,brand1))
brand2 = list(map(float,brand2))
brand3 = list(map(float,brand3))
brand4 = list(map(float,brand4))

Brand = ['Apple','XBOX','Mar','','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure()
plt.plot(np.arange(6),brand1, label='AppleTV4')
plt.plot(np.arange(6),brand2, label='AppleTV5')

plt.legend()
plt.xlabel('count',size=18)
plt.ylabel('duration $^\circ$C', size=18)

plt.savefig('test.png',dpi=300)

plt.show()
