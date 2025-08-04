import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

#time_slots = 1000
#time_slots = 500
time_slots = 400
dqn_score = np.load('dqn_scores.npy')
ddqn_score = np.load('ddqn_scores.npy')
drqn_score = np.load('drqn_scores.npy')
ppo_score = np.load('ppo_scores.npy')
a2c_score = np.load('a2c_scores.npy')
#a2c_score = np.load('a2c_means.npy')


dqn_loss = np.load('dqn_losses.npy')
ddqn_loss = np.load('ddqn_losses.npy')
drqn_loss = np.load('drqn_losses.npy')
ppo_loss = np.load('ppo_losses.npy')
a2c_loss = np.load('a2c_losses.npy')

print(a2c_score.shape)
print(f'a2c score : {a2c_score}')
print(ppo_score.shape)
print(f'ppo score : {ppo_score}')
print(ddqn_score.shape)
print(f'ddqn score : {ddqn_score}')

print(a2c_loss.shape)
print(f'a2c loss : {a2c_loss}')
print(dqn_loss.shape)
print(f'dqn loss : {dqn_loss}')
print(ddqn_loss.shape)
print(f'ddqn loss : {ddqn_loss}')


#fig = plt.figure(figsize=(10,10)) ## 캔버스 생성
#fig.set_facecolor('white') ## 캔버스 색상 설정

#plt.title('Scores/Losses DRL models', fontsize=15)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(10,10), facecolor='white')
fig.suptitle('Scores & Losses of DRL Models', fontsize=18, y=0.98)

#ax = fig.add_subplot(211) ## 그림 뼈대(프레임) 생성, 2x1 matrix 의 1행
#plt.xlabel('Time Slot')
#plt.ylabel('Average Scores')
ax.set_title('Average Scores over Time', fontsize=14, pad=10)

#bx = fig.add_subplot(212) # 2x1 matrix 의 2행
#plt.xlabel('Time Slot')
#plt.ylabel('Average Losses')
bx.set_title('Average Losses over Time', fontsize=14, pad=10)

ax.plot(np.arange(time_slots), dqn_score, marker='+',label='DQN score', markersize=5, markevery=10)
ax.plot(np.arange(time_slots), drqn_score, marker='D',label='DRQN score', markersize=5, markevery=10)
ax.plot(np.arange(time_slots), a2c_score, marker='v',label='A2C score', markersize=5, markevery=10) ## 선그래프 생성
ax.plot(np.arange(400), ppo_score, marker='o',label='PPO score', markersize=5, markevery=10) ## 선그래프 생성
ax.plot(np.arange(400), ddqn_score, marker='h',label='DDQN score', markersize=5, markevery=10) ## 선그래프 생성


bx.plot(np.arange(time_slots), dqn_loss, marker='v',label='DQN loss', markersize=3, markevery=10) ## 선그래프 생성
bx.plot(np.arange(time_slots), drqn_loss, marker='*',label='DRQN loss', markersize=3, markevery=10) ## 선그래프 생성
bx.plot(np.arange(time_slots), a2c_loss, marker='.',label='A2C loss', markersize=3, markevery=10) ## 선그래프 생성
bx.plot(np.arange(400), ppo_loss, marker='_',label='PPO loss', markersize=3, markevery=10) ## 선그래프 생성

bx.plot(np.arange(299), ddqn_loss[:299], marker='o',label='DDQN loss', markersize=5, markevery=10) ## 선그래프 생성

#ax.grid(True, linestyle='--')
#bx.grid(True, linestyle='--')

for axis in (ax, bx):
    axis.grid(True, linestyle='--')
    axis.set_xlim(0, 400)
    axis.set_xticks(np.arange(0, 401, 50))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axis.minorticks_off()
    axis.legend()

plt.tight_layout()

## 타이틀 설정
plt.show()
