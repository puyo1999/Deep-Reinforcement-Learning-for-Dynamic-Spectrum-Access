# TD Error를 저장할 메모리 클래스
import numpy as np

TD_ERROR_EPSILON = 0.0001 # Error에 더해줄 바이어스

class TDerrorMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY   # 메모리의 최대 저장 건수
        self.memory = []           # 실제 TD Error를 저장할 메모리
        self.index = 0             # 저장 위치를 가리킬 index 변수


    def push(self, td_error):
        '''1. TD Error를 메모리에 저장'''
        if len(self.memory) < self.capacity:
            self.memory.append(None) # 메모리가 가득차지 않은 경우

        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity # 다음 저장할 위치 옮기기


    def __len__(self):
        '''2. len 함수로 현재 저장된 개수 반환'''
        return len(self.memory)

    def get_prioritized_indexes(self, batch_size):
        '''3. TD Error에 따른 확률로 index 추출'''

        # TD Error의 총 절댓값 합 계산
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory) # 각 transition마다 충분히 작은 epsiolon을 더함

        # [0, sum_absolute_td_error] 구간의 batch_size개 만큼 난수 생성
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list) # batch_size개의 생성한 난수를 오름차순으로 정렬

        # 위에서 만든 난수로 index 결정
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list: # 제일 작은 난수부터 꺼내기
            # 각 memory의 td-error 값을 더해가면서, 몇번째 index
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                    abs(self.memory[idx])+TD_ERROR_EPSILON )
                idx += 1

            # TD_ERROR_EPSILON을 더한 영향으로 index가 실제 개수를 초과했을 경우를 보정
            if idx >=len(self.memory):
                idx = len(self.memory) -1
            indexes.append(idx)

        return indexes