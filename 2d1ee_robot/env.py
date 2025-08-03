import numpy as np

class Arm1DEnv:
    def __init__(self, angle_unit=5, n_goal=10):
        # 離散化したときの基準角度
        self.angle_unit = angle_unit
        # 状態空間を離散化したときの座標リスト
        self.angle_list = np.arange(0, 360, angle_unit)
        # 設定するゴールのリスト
        self.goal_list = np.arange(0, 360, n_goal)
        # とりあえずのスタート
        self.start = 0
        # とりあえずのゴール
        self.goal = 90
        # 初期の状態
        self.state = (self.start, self.goal)
        # エージェントの行動
        self.action_space = 2 # 0: +5°, 1: -5°
        # 取りうる状態の数
        self.n_state = 360//angle_unit

    # 環境のリセット
    def reset(self):
        self.start = np.random.choice(self.angle_list)
        self.goal = np.random.choice(self.goal_list)
        self.state = (self.start, self.goal)
        return self.state

    def step(self, action):  # actionは0~8（±1調整のペア）
        x, y = self.state

        # 正方向に+5°
        if action == 0:
            x = (x+5)%360
        # 負方向に-5°
        elif action == 1:
            x = (x-5)%360

        # 状態を更新
        self.state = (x, y)

        done = False
        reward = 0.0
        # 探索コスト
        reward -= ((self.goal - x) % 360)/360

        # print(x, self.goal, (self.goal==x))
        # ゴールについたら
        if self.goal == x:
            done = True
            reward += 20.0

        return self._state_to_index(self.state), reward, done
    
    def _state_to_index(self, pos):
        start, goal = pos
        diff = (goal - start + 180) % 360 - 180  # [-180, 180] の範囲に変換
        index = int(diff // self.angle_unit) + self.n_state // 2
        return index  # 例：-180°→0番目, 0°→中央, +180°→末尾
    
    @property
    def index_state(self):
        start, goal = self.state
        diff = (goal - start + 180) % 360 - 180  # [-180, 180] の範囲に変換
        index = int(diff // self.angle_unit) + self.n_state // 2
        return index  # 例：-180°→0番目, 0°→中央, +180°→末尾
        
