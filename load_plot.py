import numpy as np
import matplotlib.pyplot as plt


#time_slots = 1000
#time_slots = 500
time_slots = 100

a2c_score = np.load('a2c_scores.npy')
#a2c_score = np.load('a2c_means.npy')
drqn_score = np.load('drqn_scores.npy')
ppo_score = np.load('ppo_scores.npy')
ddqn_score = np.load('ddqn_scores.npy')

a2c_loss = np.load('a2c_losses.npy')
drqn_loss = np.load('drqn_losses.npy')
ppo_loss = np.load('ppo_losses.npy')
ddqn_loss = np.load('ddqn_losses.npy')

print(a2c_score.shape)
print(f'a2c score : {a2c_score}')
print(ppo_score.shape)
print(f'ppo score : {ppo_score}')
print(ddqn_score.shape)
print(f'ddqn score : {ddqn_score}')

print(a2c_loss.shape)
print(f'a2c loss : {a2c_loss}')
fig = plt.figure(figsize=(10,10)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정

plt.title('Scores/Losses DRL models', fontsize=15)

ax = fig.add_subplot(211) ## 그림 뼈대(프레임) 생성, 2x1 matrix 의 1행
plt.xlabel('Time Slot')
plt.ylabel('Average Scores')

bx = fig.add_subplot(212) # 2x1 matrix 의 2행
plt.xlabel('Time Slot')
plt.ylabel('Average Losses')

ax.plot(np.arange(200), a2c_score, marker='v',label='A2C score', markersize=5, markevery=10) ## 선그래프 생성
ax.plot(np.arange(time_slots+1), drqn_score, marker='D',label='DRQN score', markersize=5, markevery=10) ## 선그래프 생성
ax.plot(np.arange(time_slots+1), ppo_score, marker='o',label='PPO score', markersize=5, markevery=10) ## 선그래프 생성
ax.plot(np.arange(300), ddqn_score, marker='h',label='DDQN score', markersize=5, markevery=10) ## 선그래프 생성

bx.plot(np.arange(200), a2c_loss, marker='.',label='A2C loss', markersize=3, markevery=10) ## 선그래프 생성
bx.plot(np.arange(time_slots), drqn_loss, marker='*',label='DRQN loss', markersize=3, markevery=10) ## 선그래프 생성
bx.plot(np.arange(time_slots), ppo_loss, marker='_',label='PPO loss', markersize=3, markevery=10) ## 선그래프 생성
bx.plot(np.arange(299), ddqn_loss, marker='o',label='DDQN loss', markersize=3, markevery=10) ## 선그래프 생성

#ax.plot(days,b_visits,marker='o',label='B')
#ax.plot(days,c_visits,marker='o',label='C')
ax.grid(True, linestyle='--')
bx.grid(True, linestyle='--')
ax.legend() ## 범례
bx.legend()

## 타이틀 설정
plt.show()
