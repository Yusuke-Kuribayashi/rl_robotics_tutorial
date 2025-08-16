import numpy as np

class Arm2DEnv:
    def __init__(self, angle_unit=1, link_length=0.5):
        # ロボットの基準角度
        self.angle_unit = angle_unit
        # ロボットのリンク長
        self.link_length = link_length
        # 初期のスタート
        self.start = np.array([0, 0])                  # (1, 2)
        # とりあえずのゴール
        self.goal = np.array([0.7, 0.3])               # (1, 2)
        # 初期の状態
        self.state = np.array([self.start, self.goal]) # (2, 2)
        # 行動空間
        self.action_space = {0: [ self.angle_unit, 0],
                             1: [-self.angle_unit, 0],
                             2: [0,  self.angle_unit],
                             3: [0, -self.angle_unit],
                             4: [self.angle_unit, self.angle_unit],
                             5: [self.angle_unit, -self.angle_unit],
                             6: [-self.angle_unit, self.angle_unit],
                             7: [-self.angle_unit, -self.angle_unit],
                             8: [0, 0]}
        # 動作の終了誤差
        self.tol = 0.1

    # 環境のリセット（初期状態と目標を設定）
    def reset(self):
        self.start = np.random.uniform(0, 360, size=2)
        goal_theta = np.random.uniform(-90, 90, size=2)
        self.goal = self.fk(goal_theta)
        self.state = np.array([self.start, self.goal])
        return self.get_state()
    
    # 行動に基づいて状態更新・報酬計算
    def step(self, action):
        joint_angles, goal_position = self.state.copy()

        # 行動による状態の変化
        joint_angles = (joint_angles +  np.array(self.action_space[action])) % 360

        # 状態を更新
        self.state = np.array([joint_angles, goal_position])

        done = False
        reward = 0.0
        hand_position = self.fk(joint_angles)
        distance = np.linalg.norm(goal_position-hand_position)*0.5
        # 報酬
        # 距離ペナルティ
        reward -= distance
        # 探索ペナルティ
        reward -= 0.01

        if distance < self.tol:
            done = True
            reward += 30.0

        return self.get_state(), reward, done

    # NNに渡すための状態ベクトルを返す  
    def get_state(self):
        # 現在の関節角度(theta1, theta2)
        joint_angles = self.state[0]
        # 角度を正規化
        norm_state = (joint_angles % 360)/360.0

        # 目標となる手先座標
        goal_position = self.state[1]
        # NNに渡すために、１次元に変換
        state_vector = np.concatenate([norm_state, goal_position])
        return state_vector.astype(np.float32)
             
    def fk(self, angles):
        theta1, theta2 = np.deg2rad(angles[0]), np.deg2rad(angles[1])
        x = self.link_length*np.cos(theta1) + self.link_length*np.cos(theta1+theta2)
        y = self.link_length*np.sin(theta1) + self.link_length*np.sin(theta1+theta2)
        return np.array([x, y])
    
    def step_sim(self, action):
        joint_angles, goal_position = self.state.copy()

        # 行動による状態の変化
        joint_angles = (joint_angles +  np.array(self.action_space[action])) % 360

        # 状態を更新
        self.state = np.array([joint_angles, goal_position])

        done = False
        hand_position = self.fk(joint_angles)
        distance = np.linalg.norm(goal_position-hand_position)
        if distance < self.tol:
            done = True

        # 関節座標を返す
        x = self.link_length * np.cos(np.deg2rad(joint_angles[0]))
        y = self.link_length * np.sin(np.deg2rad(joint_angles[0]))
        elbow_xy = (x, y)

        return hand_position, elbow_xy, done
