# ============================================
# DQN (Deep Q-Network) — Scratch Template (comments only)
# ============================================
# このファイルは「穴埋め形式」で DQN をスクラッチ実装するための
# コメントだけのテンプレートです。必要箇所を TODO に従って実装してください。
# 実装順序は上から埋めていくとスムーズです。
# --------------------------------------------

# 0) 必要ライブラリのインポート
# TODO: numpy, random, collections.deque, torch（nn, optim, no_grad など）をインポート
# 例: import numpy as np / import torch / import torch.nn as nn ...

# 1) 環境の準備
# TODO: 自作環境 Arm2DEnv をインポート or 同ファイルに定義
# 必要メソッド: reset() -> state, step(action) -> (next_state, reward, done), get_state() -> state_vec
# 状態 shape: (state_dim,) / 行動数: action_dim

# 2) ハイパーパラメータの設定
# TODO: 以下を定義
# - NUM_EPISODES: 総エピソード数
# - MAX_STEPS: 1エピソードあたりの最大ステップ数
# - BATCH_SIZE: 学習に使うミニバッチサイズ
# - GAMMA: 割引率 (0.99 など)
# - LR: 学習率 (1e-3 など)
# - EPSILON_START, EPSILON_END, EPSILON_DECAY: ε-greedy の各パラメータ
# - TARGET_UPDATE_INTERVAL: ターゲットネットワークの更新間隔（エピソード or ステップ）
# - REPLAY_CAPACITY: リプレイバッファ容量

# 3) DQN モデルの定義
# TODO: nn.Module を継承したクラス DQN を作り、
# __init__(input_dim, output_dim) と forward(x) を実装
# 推奨: 全結合 2〜3層 (例: 128 -> 128) + ReLU / 出力は action_dim ノード

# 4) ターゲットネットワークの準備
# TODO: policy_net と target_net を同じ構造で初期化し、
# target_net に policy_net の state_dict をコピー
# target_net は学習で更新せず、一定間隔で同期（ハード更新 or ソフト更新）

# 5) Optimizer の用意
# TODO: torch.optim.Adam 等で policy_net のパラメータを最適化するオプティマイザを作成

# 6) Replay Buffer の実装
# TODO: class ReplayBuffer(capacity):
#   - 内部に deque(maxlen=capacity) を保持
#   - push(state, action, reward, next_state, done): 経験を追加
#   - sample(batch_size): ランダムサンプリングして (states, actions, rewards, next_states, dones) を返す
#   - __len__: 現在サイズを返す

# 7) ε-greedy による行動選択関数
# TODO: select_action(model, state, epsilon, action_dim):
#   - 乱数 < epsilon ならランダム行動
#   - それ以外は model(state) の argmax を選択
#   - state は torch.FloatTensor に変換して forward / with torch.no_grad()

# 8) 環境・ネットワーク・バッファの初期化
# TODO:
#   - env = Arm2DEnv()
#   - state_dim = env.get_state().shape[0]
#   - action_dim = len(env.action_space)
#   - policy_net = DQN(state_dim, action_dim)
#   - target_net = DQN(state_dim, action_dim) -> policy から重みコピー
#   - optimizer = Adam(..., lr=LR)
#   - buffer = ReplayBuffer(REPLAY_CAPACITY)
#   - epsilon = EPSILON_START

# 9) 学習ループ（エピソード単位）
# for episode in range(NUM_EPISODES):
#   - state = env.reset()
#   - episode_reward = 0
#   - for step in range(MAX_STEPS):
#       (1) 行動選択: action = select_action(policy_net, state, epsilon, action_dim)
#       (2) 環境遷移: next_state, reward, done = env.step(action)
#       (3) バッファ保存: buffer.push(state, action, reward, next_state, done)
#       (4) 状態更新: state = next_state / 報酬加算: episode_reward += reward
#       (5) 学習: if len(buffer) >= BATCH_SIZE -> optimize_step(...)
#       (6) 終了処理: if done: break
#   - εの減衰: epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
#   - ターゲット更新（ハード）: if episode % TARGET_UPDATE_INTERVAL == 0: target <- policy
#   - ログ出力: Episode, episode_reward, epsilon

# 10) optimize_step 関数（学習 1 ステップ）
# TODO: def optimize_step(policy_net, target_net, optimizer, buffer, batch_size, gamma):
#   - バッファからサンプル: states, actions, rewards, next_states, dones = buffer.sample(batch_size)
#   - torch.Tensor へ変換 / shape 整形（actions, rewards, dones は (B,1) になるように）
#   - 現在の Q(s,a): q_values = policy_net(states).gather(1, actions)
#   - 次状態の max Q(s',a')（target_net を使用、no_grad）: next_q = target_net(next_states).max(dim=1)[0].unsqueeze(1)
#   - TD ターゲット: target_q = rewards + gamma * next_q * (1 - dones)
#   - 損失: MSE(q_values, target_q)
#   - 逆伝播: optimizer.zero_grad(); loss.backward(); optimizer.step()
#   - 戻り値: loss.item()（ログ用）

# 11) ターゲットネット更新（ソフト更新を使う場合）
# TODO: soft_update(target_net, policy_net, tau):
#   for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
#       target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
# ※ ハード更新とどちらか片方を使用

# 12) 学習ログ・モデル保存（任意）
# TODO:
#   - 各エピソードの total reward をリストで保存し、matplotlib で可視化
#   - best モデルの保存（torch.save(policy_net.state_dict(), 'dqn.pt')）
#   - 学習後の評価ループ（ε=0 で greedy に動かして可視化）

# 13) 実行スクリプト部
# TODO: if __name__ == "__main__":
#   - 上記の初期化と学習ループを呼び出す
#   - ログや可視化を行う
