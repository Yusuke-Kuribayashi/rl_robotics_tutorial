import pygame
import math

class ArmViewer:
    """
    純粋に「描画と座標変換」だけ担当するビューワ
    - 世界座標 (x,y) を受け取り、ピクセルに描画
    - クリック→世界座標へ変換（主処理側で env.set_goal を呼ぶ）
    """
    def __init__(self, width=700, height=700, world_radius_max=1.0, title="2DOF Arm (DQN)"):
        pygame.init()
        self.W, self.H = width, height
        self.cx, self.cy = width // 2, height // 2
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont(None, 24)
        # スケール: ワールド半径1.0 → 画面の 0.45*min(W,H)
        self.scale = 0.45 * min(width, height) / float(world_radius_max)
        self.clock = pygame.time.Clock()

    # ---- 座標変換 ----
    def world_to_pix(self, x, y):
        # 右＋、上＋の世界 → 画面（右＋、下＋）
        X = int(self.cx + x * self.scale)
        Y = int(self.cy - y * self.scale)
        return X, Y

    def pix_to_world(self, X, Y):
        x = (X - self.cx) / self.scale
        y = (self.cy - Y) / self.scale
        return x, y

    # ---- 描画 ----
    def render(self, base_xy, elbow_xy, hand_xy, goal_xy, info_lines=None):
        self.screen.fill((255, 255, 255))
        # ワークスペース
        pygame.draw.circle(self.screen, (220, 220, 255), self.world_to_pix(0, 0), 1.0*self.scale)

        # アーム
        bX, bY = self.world_to_pix(*base_xy)
        eX, eY = self.world_to_pix(*elbow_xy)
        hX, hY = self.world_to_pix(*hand_xy)
        pygame.draw.circle(self.screen, (0, 0, 0), (bX, bY), 6)
        pygame.draw.circle(self.screen, (0, 0, 0), (eX, eY), 6)
        pygame.draw.line(self.screen, (0, 0, 200), (bX, bY), (eX, eY), 8)
        pygame.draw.line(self.screen, (0, 0, 255), (eX, eY), (hX, hY), 8)

        # 目標 & 手先
        gX, gY = self.world_to_pix(*goal_xy)
        pygame.draw.circle(self.screen, (0, 180, 0), (gX, gY), 7)
        pygame.draw.circle(self.screen, (255, 0, 0), (hX, hY), 7)

        # 情報テキスト
        if info_lines:
            for i, text in enumerate(info_lines):
                self.screen.blit(self.font.render(text, True, (0, 0, 0)), (10, 10 + 24*i))

        pygame.display.flip()

    def tick(self, fps=30):
        self.clock.tick(fps)

    def set_goal(self, pos):
        x, y = self.pix_to_world(*pos)
        # print(f"World coordinates: ({x:.2f}, {y:.2f}), Pixel coordinates: {pos}")
        return (x, y)

if __name__ == "__main__":
    viewer = ArmViewer()
    # 例: 座標変換と描画のテスト
    viewer.render((0, 0), (0.5, 0), (1, 0), (0.5, 1),
                  info_lines=["Test Arm Viewer", "Click to set goal"])
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                viewer.print_pose(pygame.mouse.get_pos())
        viewer.tick()
        pygame.time.delay(100)
    pygame.quit()
