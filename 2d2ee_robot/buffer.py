import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, batch_size, capacity=10_000):
        # ストックしておく行動量の定義
        self.buffer = deque(maxlen=capacity)
        # バッチサイズの設定
        self.batch_size = batch_size

    # 行動を保存
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # ランダムに行動を取得
    def sample(self):
        batch_data = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch_data))

        return states, actions, rewards, next_states, dones
    
    # 現在の保存データ数を返す
    def __len__(self):
        return len(self.buffer)