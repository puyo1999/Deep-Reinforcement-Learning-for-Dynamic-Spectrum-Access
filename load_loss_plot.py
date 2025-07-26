import numpy as np
import matplotlib.pyplot as plt

time_slots = 800
t1 = np.arange(0, time_slots, 1)

actor_loss = np.load('actor_losses.npy')
critic_loss = np.load('critic_losses.npy')

print(actor_loss.shape)
print(f'actor_loss : {actor_loss}')
print(critic_loss.shape)
print(f'critic_loss : {critic_loss}')

fig = plt.figure(figsize=(15,10)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정

al = fig.add_subplot(211) ## 그림 뼈대(프레임) 생성, 2x1 matrix 의 1행
plt.ylabel('Actor loss')

cl = fig.add_subplot(212) # 2x1 matrix 의 2행
plt.xlabel('Time Slot')
plt.ylabel('Critic loss')

plt.title('Actor vs Critic Loss Over Time', fontsize=15)
# actor loss
al.plot(
    t1, actor_loss,
    'rv-',                      # 빨간색 삼각형 마커
    markersize=3,
    markevery=10,
    linewidth=1,
    label='actor loss'
)

# critic loss
cl.plot(
    t1, critic_loss,
    'bD-',                      # 파란색 다이아몬드 마커
    markersize=3,
    markevery=10,
    linewidth=1,
    label='critic loss'
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()