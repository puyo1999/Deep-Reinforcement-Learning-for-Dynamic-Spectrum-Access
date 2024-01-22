import numpy as np
import matplotlib.pyplot as plt

time_slots = 1000
a2c_score = np.load('a2c_scores.npy')
drqn_score = np.load('drqn_scores.npy')

a2c_loss = np.load('a2c_losses.npy')
drqn_loss = np.load('drqn_losses.npy')

print(a2c_score.shape)
print(f'a2c score : {a2c_score}')
fig = plt.figure(figsize=(10,10)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정

plt.title('Scores/Losses DRL models', fontsize=15)

ax = fig.add_subplot(211) ## 그림 뼈대(프레임) 생성
plt.xlabel('Time Slot')
plt.ylabel('Average Scores')

bx = fig.add_subplot(212)
plt.xlabel('Time Slot')
plt.ylabel('Average Losses')

ax.plot(np.arange(time_slots+1), a2c_score, marker='.',label='A2C score', markersize=1.5) ## 선그래프 생성
ax.plot(np.arange(time_slots+1), drqn_score, marker='.',label='DRQN score', markersize=1.5) ## 선그래프 생성

bx.plot(np.arange(time_slots), a2c_loss, marker=',',label='A2C loss', markersize=1.5) ## 선그래프 생성
bx.plot(np.arange(time_slots), drqn_loss, marker=',',label='DRQN loss', markersize=1.5) ## 선그래프 생성

#ax.plot(days,b_visits,marker='o',label='B')
#ax.plot(days,c_visits,marker='o',label='C')
ax.legend() ## 범례
bx.legend()

## 타이틀 설정
plt.show()
