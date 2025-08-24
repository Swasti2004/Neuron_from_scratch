import pygame
import numpy as np

pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Logic Gate Perceptron Visualizer")

# Colors
WHITE = (250, 250, 250)
BLACK = (40, 40, 40)
GREEN = (0, 200, 0)
RED = (220, 60, 60)
YELLOW = (250, 210, 50)
BLUE = (80, 120, 255)
LIGHT_GREY = (230, 230, 230)
DARK_GREY = (100, 100, 100)

# Fonts
font_small = pygame.font.SysFont("arial", 22)
font_big = pygame.font.SysFont("arial", 48, bold=True)
font_mid = pygame.font.SysFont("arial", 28, bold=True)

# Perceptron params
weights = [1.0, 1.0]
bias = -1.5
activation = lambda x: 1 if x >= 0 else 0

# Inputs
x1, x2 = 0, 0

# Positions
input_pos = [(200, 200), (200, 350)]
hidden_pos = (450, 275)
output_pos = (700, 275)

# Gates
gates = {
    "AND": [1.0, 1.0, -1.5],
    "OR": [1.0, 1.0, -0.5],
    "NAND": [-1.0, -1.0, 1.5],
    "XOR": [1.0, 1.0, -1.0]  # won't work properly
}
gate_names = list(gates.keys())
gate_index = 0

# Drag states
dragging_weight = None
dragging_bias = False

def perceptron_output(x1, x2, w, b):
    return activation(w[0]*x1 + w[1]*x2 + b)

def draw_glowing_node(pos, value, label=None, active=False, highlight=False):
    # Glow effect if active
    glow_color = BLUE if highlight else DARK_GREY
    for r in range(40, 60, 5):
        pygame.draw.circle(screen, glow_color, pos, r, 2)

    color = GREEN if value == 1 and active else (BLACK if not active else WHITE)
    pygame.draw.circle(screen, color, pos, 35)
    pygame.draw.circle(screen, BLUE if highlight else BLACK, pos, 35, 3)

    text = font_mid.render(str(value), True, WHITE if value == 1 else BLACK)
    screen.blit(text, (pos[0]-10, pos[1]-15))

    if label:
        lbl = font_small.render(label, True, BLACK)
        screen.blit(lbl, (pos[0]-15, pos[1]+45))

def draw_connection(p1, p2, weight):
    color = GREEN if weight >= 0 else RED
    thickness = max(2, int(abs(weight)*2))
    pygame.draw.line(screen, color, p1, p2, thickness)

    midx, midy = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
    rect = pygame.draw.rect(screen, YELLOW, (midx-25, midy-20, 50, 25), border_radius=6)
    text = font_small.render(f"{weight:.2f}", True, BLACK)
    screen.blit(text, (midx-20, midy-17))
    return rect

def draw_panel():
    # Side panel
    pygame.draw.rect(screen, LIGHT_GREY, (800, 0, 200, HEIGHT))
    pygame.draw.line(screen, DARK_GREY, (800, 0), (800, HEIGHT), 3)

    # Title
    title = font_big.render(gate_names[gate_index], True, BLUE)
    screen.blit(title, (820, 20))

    # Truth table
    table_title = font_mid.render("Truth Table", True, BLACK)
    screen.blit(table_title, (830, 90))

    truth = {
        "AND": [(0,0,0),(0,1,0),(1,0,0),(1,1,1)],
        "OR":  [(0,0,0),(0,1,1),(1,0,1),(1,1,1)],
        "NAND":[(0,0,1),(0,1,1),(1,0,1),(1,1,0)],
        "XOR": [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
    }

    for i,(a,b,expected) in enumerate(truth[gate_names[gate_index]]):
        y = 130 + i*40
        out = perceptron_output(a,b,weights,bias)
        text = font_small.render(f"{a} {b} → {out}", True, BLACK)
        screen.blit(text, (830, y))
        mark = "✔" if out==expected else "✘"
        mark_color = GREEN if mark=="✔" else RED
        mtxt = font_small.render(mark, True, mark_color)
        screen.blit(mtxt, (950, y))

def draw_instructions():
    lines = [
        "Controls:",
        "Click input to toggle",
        "Drag weight labels ↑↓",
        "Drag bias node ↑↓",
        "Press SPACE → switch gate"
    ]
    for i,l in enumerate(lines):
        txt = font_small.render(l, True, DARK_GREY)
        screen.blit(txt, (50, 500+i*20))

running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)

    out = perceptron_output(x1, x2, weights, bias)

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx,my = event.pos
            for i,pos in enumerate(input_pos):
                if (mx-pos[0])**2+(my-pos[1])**2 <= 35**2:
                    if i==0: x1 = 1-x1
                    else: x2 = 1-x2

            if w1_rect.collidepoint(mx,my): dragging_weight = 0
            elif w2_rect.collidepoint(mx,my): dragging_weight = 1
            elif (mx-hidden_pos[0])**2+(my-hidden_pos[1])**2 <= 35**2: dragging_bias = True

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_weight = None
            dragging_bias = False

        elif event.type == pygame.MOUSEMOTION:
            if dragging_weight is not None:
                weights[dragging_weight] += event.rel[1]*-0.01
            if dragging_bias:
                bias += event.rel[1]*-0.01

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                gate_index = (gate_index+1)%len(gate_names)
                w1,w2,b = gates[gate_names[gate_index]]
                weights=[w1,w2]; bias=b

    # Draw
    w1_rect = draw_connection(input_pos[0], hidden_pos, weights[0])
    w2_rect = draw_connection(input_pos[1], hidden_pos, weights[1])
    draw_connection(hidden_pos, output_pos, 1.0)

    draw_glowing_node(input_pos[0], x1, "X1", active=True)
    draw_glowing_node(input_pos[1], x2, "X2", active=True)
    draw_glowing_node(hidden_pos, round(bias,2), "Bias", active=False)
    draw_glowing_node(output_pos, out, "Out", active=True, highlight=True)

    draw_panel()
    draw_instructions()

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
