import pygame
import sys
import math
import numpy as np
import time

class ArmVisualizer:
    def __init__(self, q_table, angle_unit, n_state, arm_length=150, width=600, height=600, goal_interval=10):
        pygame.init()
        self.ARM_LENGTH = arm_length
        self.WIDTH = width
        self.HEIGHT = height
        self.CENTER_X = width // 2
        self.CENTER_Y = height // 2

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("1自由度ロボットアーム 可視化")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()

        self.angle = 0.0                # 現在の関節角度
        self.target_angle = self.angle  # 現在選ばれている目標角度
        self.target_angles = np.arange(0, 360, goal_interval)  # ゴール候補のリスト
        self.running = True

        self.q_table = q_table
        self.angle_unit = angle_unit
        self.n_state = n_state

    def draw(self):
        self.screen.fill((255, 255, 255))

        # アームの可動円
        pygame.draw.arc(
            self.screen, (220, 220, 255),
            (self.CENTER_X - self.ARM_LENGTH, self.CENTER_Y - self.ARM_LENGTH,
             self.ARM_LENGTH * 2, self.ARM_LENGTH * 2),
            0, 2 * math.pi, width=20
        )

        # 各ゴール候補を描画
        for angle in self.target_angles:
            rad = math.radians(angle)
            gx = self.CENTER_X + self.ARM_LENGTH * math.cos(rad)
            gy = self.CENTER_Y - self.ARM_LENGTH * math.sin(rad)
            pygame.draw.circle(self.screen, (0, 180, 0), (int(gx), int(gy)), 6)

        # 現在の目標角度（ハイライト）
        if self.target_angle is not None:
            rad = math.radians(self.target_angle)
            gx = self.CENTER_X + self.ARM_LENGTH * math.cos(rad)
            gy = self.CENTER_Y - self.ARM_LENGTH * math.sin(rad)
            pygame.draw.circle(self.screen, (0, 255, 0), (int(gx), int(gy)), 10)

        # アームの描画
        rad = math.radians(self.angle)
        end_x = self.CENTER_X + self.ARM_LENGTH * math.cos(rad)
        end_y = self.CENTER_Y - self.ARM_LENGTH * math.sin(rad)
        pygame.draw.line(self.screen, (0, 0, 255),
                         (self.CENTER_X, self.CENTER_Y), (end_x, end_y), 6)
        pygame.draw.circle(self.screen, (0, 0, 0), (self.CENTER_X, self.CENTER_Y), 5)
        pygame.draw.circle(self.screen, (255, 0, 0), (int(end_x), int(end_y)), 6)

        # ラベル
        if self.target_angle is not None:
            label = self.font.render(f"target angle: {self.target_angle:.1f}°", True, (0, 100, 0))
            self.screen.blit(label, (10, 10))
        label2 = self.font.render(f"now angle: {self.angle:.1f}°", True, (0, 0, 0))
        self.screen.blit(label2, (10, 40))

        pygame.display.flip()

    def pos_to_angle(self, x, y):
        dx = x - self.CENTER_X
        dy = self.CENTER_Y - y
        return math.degrees(math.atan2(dy, dx)) % 360

    def set_angle(self, angle):
        self.angle = angle % 360

    def set_target_from_click(self, pos):
        """クリック位置から一番近いゴール候補を選ぶ"""
        clicked_angle = self.pos_to_angle(*pos)
        closest_angle = min(self.target_angles, key=lambda a: abs(a - clicked_angle))
        self.target_angle = closest_angle
        # print(f"クリック -> 目標角度: {self.target_angle:.1f}°")

    def step(self):
        # q_tableから最適方策を選択
        action = np.argmax(self.q_table[self.state_to_index()])
        
        # 正方向に+5°
        if action == 0:
            self.angle = (self.angle+5)%360
        # 負方向に-5°
        elif action == 1:
            self.angle = (self.angle-5)%360

    def state_to_index(self):
        diff = (self.target_angle - self.angle + 180) % 360 - 180
        index = int(diff // self.angle_unit) + self.n_state // 2
        return index 
    
    def run(self):
        """メインループ"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.set_target_from_click(pygame.mouse.get_pos())

            if self.angle != self.target_angle:
                self.step()

            self.draw()
            self.clock.tick(30)
            time.sleep(0.05)

        pygame.quit()
        sys.exit()
