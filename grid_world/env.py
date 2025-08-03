import numpy as np

class GridEnv:
    def __init__(self, size=4):
        self.size = size
        # 左上をスタート
        self.start = (0, 0)
        # 右下をゴール
        self.goal = (size - 1, size - 1)
        # 通れない道を設定
        self.holes = {(1, 1), (2, 3), (3, 1)}
        # 初期の状態をスタート地点とする
        self.state = self.start
        # エージェントの行動
        self.action_space = 4  # 0:上, 1:右, 2:下, 3:左
        # 観測環境
        self.observation_space = size * size

    # 環境をリセットさせる
    def reset(self):
        self.state = self.start
        return self._state_to_index(self.state)

    # 行動を行ったときに、エージェントがどの状態になるかを定義
    # 出力としては、観測値・報酬・試行の終了の有無
    def step(self, action):
        x, y = self.state

        # マップ範囲外であるか確認
        if action == 0:  # 上
            x = max(0, x - 1)
        elif action == 1:  # 右
            y = min(self.size - 1, y + 1)
        elif action == 2:  # 下
            x = min(self.size - 1, x + 1)
        elif action == 3:  # 左
            y = max(0, y - 1)

        # 状態を更新
        self.state = (x, y)

        done = False
        reward = 0.0

        # 穴に落ちたら終了
        if self.state in self.holes:
            reward -= 5.0
            done = True
        # ゴールについたら、報酬を渡す
        elif self.state == self.goal:
            done = True
            reward += 5.0

        reward-= 0.5

        return self._state_to_index(self.state), reward, done

    # 現在の状態を観測に変更
    def _state_to_index(self, pos):
        return pos[0] * self.size + pos[1]

    # 観測値からstateに変更
    def _index_to_state(self, idx):
        return (idx // self.size, idx % self.size)
    
    @property
    def index_state(self):
        return self.state[0]*self.size + self.state[1]

