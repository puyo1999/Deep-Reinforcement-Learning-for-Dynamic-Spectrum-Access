import numpy as np
import matplotlib.pyplot as plt

time_slots = 1000
a2c_score = np.load('a2c_scores.npy')
drqn_score = np.load('drqn_scores.npy')
print(a2c_score.shape)
print(f'a2c score : {a2c_score}')
fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성
ax.plot(np.arange(time_slots), a2c_score, marker='o',label='A2C', markersize=3) ## 선그래프 생성
ax.plot(np.arange(time_slots), drqn_score, marker='o',label='DRQN', markersize=3) ## 선그래프 생성

#ax.plot(days,b_visits,marker='o',label='B')
#ax.plot(days,c_visits,marker='o',label='C')
ax.legend() ## 범례
plt.title('Scores for DRL models', fontsize=20)
## 타이틀 설정
plt.show()
