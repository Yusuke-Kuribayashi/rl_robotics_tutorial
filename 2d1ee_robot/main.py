import pygame
import sys
import math

ARM_LENGTH = 150
CENTER_X = 300
CENTER_Y = 300
WIDTH, HEIGHT = 600, 600

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2次元1自由度ロボットアーム")
font = pygame.font.SysFont(None, 24)

angle = 0
clock = pygame.time.Clock()

input_active = False
user_input = "0"


def draw_arm(angle):
    screen.fill((255, 255, 255))

    # 可動範囲（180度扇形）
    pygame.draw.arc(
        screen,
        (173, 216, 230),
        (CENTER_X - ARM_LENGTH, CENTER_Y - ARM_LENGTH, ARM_LENGTH * 2, ARM_LENGTH * 2),
        0, 2*math.pi,
        width=20
    )

    rad = math.radians(angle)
    end_x = CENTER_X + ARM_LENGTH * math.cos(rad)
    end_y = CENTER_Y - ARM_LENGTH * math.sin(rad)

    # アーム
    pygame.draw.line(screen, (0, 0, 255), (CENTER_X, CENTER_Y), (end_x, end_y), 6)
    # 基点
    pygame.draw.circle(screen, (0, 0, 0), (CENTER_X, CENTER_Y), 5)
    # 先端
    pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 6)

    # 入力UI
    label = font.render("角度 [Enterで決定]:", True, (0, 0, 0))
    screen.blit(label, (10, 10))
    input_box = font.render(user_input, True, (0, 0, 0))
    pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(150, 5, 60, 25))
    screen.blit(input_box, (155, 10))

    pygame.display.flip()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                try:
                    angle = float(user_input)
                except ValueError:
                    angle = 0
                user_input = ""

            elif event.key == pygame.K_BACKSPACE:
                user_input = user_input[:-1]

            elif event.unicode.isdigit() or event.unicode == '-' or event.unicode == '.':
                user_input += event.unicode

    draw_arm(angle)
    clock.tick(30)
