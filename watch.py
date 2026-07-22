import numpy as np
import torch
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from collections import deque
from env import make_env, OBS_SIZE, ACT_SIZE
from collect import ActorCritic
import math

BLUE_COLOR = (0.2, 0.6, 1.0)
ORANGE_COLOR = (1.0, 0.5, 0.1)
BALL_COLOR = (1.0, 1.0, 1.0)

FIELD_LENGTH = 10280
FIELD_WIDTH = 8192
BALL_RADIUS = 92
GOAL_HEIGHT = 642
GOAL_DEPTH = 880
GOAL_WIDTH = 1786

CAM_NAMES = ["Broadcast", "Chase Blue", "Chase Orange", "Top Down", "Blue Goal", "Orange Goal"]


def draw_sphere(x, y, z, radius, color, slices=20, stacks=14):
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(x, y, z)
    quad = gluNewQuadric()
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)
    glPopMatrix()


def draw_box(x, y, z, sx, sy, sz, color, yaw=0):
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(math.degrees(yaw), 0, 1, 0)
    hx, hy, hz = sx/2, sy/2, sz/2
    glBegin(GL_QUADS)
    for normal, verts in [
        ((0,0,1), [(-hx,-hy,hz),(hx,-hy,hz),(hx,hy,hz),(-hx,hy,hz)]),
        ((0,0,-1), [(-hx,-hy,-hz),(-hx,hy,-hz),(hx,hy,-hz),(hx,-hy,-hz)]),
        ((0,1,0), [(-hx,hy,-hz),(-hx,hy,hz),(hx,hy,hz),(hx,hy,-hz)]),
        ((0,-1,0), [(-hx,-hy,-hz),(hx,-hy,-hz),(hx,-hy,hz),(-hx,-hy,hz)]),
        ((1,0,0), [(hx,-hy,-hz),(hx,hy,-hz),(hx,hy,hz),(hx,-hy,hz)]),
        ((-1,0,0), [(-hx,-hy,-hz),(-hx,-hy,hz),(-hx,hy,hz),(-hx,hy,-hz)]),
    ]:
        glNormal3f(*normal)
        for v in verts:
            glVertex3f(*v)
    glEnd()
    glPopMatrix()


def draw_nose(x, y, z, yaw, color, size=60):
    nx = x + math.sin(yaw) * size
    nz = z + math.cos(yaw) * size
    glColor3f(*color)
    glPointSize(6)
    glBegin(GL_POINTS)
    glVertex3f(nx, y + 20, nz)
    glEnd()


def draw_field():
    hw = FIELD_WIDTH / 2
    hl = FIELD_LENGTH / 2

    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glColor3f(*((0.4, 0.4, 0.5)))
    glVertex3f(-hw, 1, -hl)
    glVertex3f(hw, 1, -hl)
    glVertex3f(hw, 1, hl)
    glVertex3f(-hw, 1, hl)
    glEnd()

    glBegin(GL_LINES)
    glVertex3f(-hw, 1, 0)
    glVertex3f(hw, 1, 0)
    glEnd()

    glBegin(GL_LINE_LOOP)
    for i in range(48):
        angle = 2 * math.pi * i / 48
        glVertex3f(500 * math.cos(angle), 1, 500 * math.sin(angle))
    glEnd()

    gw = GOAL_WIDTH / 2
    gd = GOAL_DEPTH

    glLineWidth(3)
    glBegin(GL_LINES)
    glColor3f(0.1, 0.3, 0.9)
    glVertex3f(-gw, 0, -hl)
    glVertex3f(-gw, GOAL_HEIGHT, -hl)
    glVertex3f(gw, 0, -hl)
    glVertex3f(gw, GOAL_HEIGHT, -hl)
    glVertex3f(-gw, GOAL_HEIGHT, -hl)
    glVertex3f(gw, GOAL_HEIGHT, -hl)
    glVertex3f(-gw, 0, -hl - gd)
    glVertex3f(-gw, GOAL_HEIGHT, -hl - gd)
    glVertex3f(gw, 0, -hl - gd)
    glVertex3f(gw, GOAL_HEIGHT, -hl - gd)
    glVertex3f(-gw, GOAL_HEIGHT, -hl - gd)
    glVertex3f(gw, GOAL_HEIGHT, -hl - gd)
    glVertex3f(-gw, GOAL_HEIGHT, -hl)
    glVertex3f(-gw, GOAL_HEIGHT, -hl - gd)
    glVertex3f(gw, GOAL_HEIGHT, -hl)
    glVertex3f(gw, GOAL_HEIGHT, -hl - gd)
    glEnd()

    glBegin(GL_LINES)
    glColor3f(0.9, 0.3, 0.1)
    glVertex3f(-gw, 0, hl)
    glVertex3f(-gw, GOAL_HEIGHT, hl)
    glVertex3f(gw, 0, hl)
    glVertex3f(gw, GOAL_HEIGHT, hl)
    glVertex3f(-gw, GOAL_HEIGHT, hl)
    glVertex3f(gw, GOAL_HEIGHT, hl)
    glVertex3f(-gw, 0, hl + gd)
    glVertex3f(-gw, GOAL_HEIGHT, hl + gd)
    glVertex3f(gw, 0, hl + gd)
    glVertex3f(gw, GOAL_HEIGHT, hl + gd)
    glVertex3f(-gw, GOAL_HEIGHT, hl + gd)
    glVertex3f(gw, GOAL_HEIGHT, hl + gd)
    glVertex3f(-gw, GOAL_HEIGHT, hl)
    glVertex3f(-gw, GOAL_HEIGHT, hl + gd)
    glVertex3f(gw, GOAL_HEIGHT, hl)
    glVertex3f(gw, GOAL_HEIGHT, hl + gd)
    glEnd()

    glLineWidth(1)
    glBegin(GL_POINTS)
    glColor3f(1, 0.8, 0)
    pads = [
        (0, 0), (-1792, 0), (1792, 0), (-3584, 0), (3584, 0),
        (0, -1792), (0, 1792), (0, -3584), (0, 3552),
        (-1792, -1792), (1792, -1792), (-1792, 1792), (1792, 1792),
        (-3584, -1792), (3584, -1792), (-3584, 1792), (3584, 1792),
    ]
    for px, pz in pads:
        glVertex3f(px, 2, pz)
    glEnd()


def draw_trail(points, color):
    if len(points) < 2:
        return
    glLineWidth(2)
    glBegin(GL_LINE_STRIP)
    glColor4f(*color, 0.5)
    for p in points:
        glVertex3f(p[0], 2, p[1])
    glEnd()


def render_hud(x, y, text, font_obj, color=(255, 255, 255)):
    surface = font_obj.render(text, True, color, (0, 0, 0))
    data = pygame.image.tostring(surface, "RGBA", True)
    w, h = surface.get_size()
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 1280, 720, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x, y)
    glTexCoord2f(1, 1); glVertex2f(x + w, y)
    glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
    glTexCoord2f(0, 0); glVertex2f(x, y + h)
    glEnd()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glDeleteTextures([tex_id])


def get_camera(cam_id, blue_pos, orange_pos, ball_pos):
    bx, by, bz = blue_pos
    ox, oy, oz = orange_pos
    bpx, bpy, bpz = ball_pos
    mid_x = (bx + ox) / 2
    mid_z = (bz + oz) / 2

    if cam_id == 0:
        eye_x = 0
        eye_y = 12000
        eye_z = -15000
        return eye_x, eye_y, eye_z, 0, 0, 0
    elif cam_id == 1:
        eye_x = bx
        eye_y = by + 500
        eye_z = bz - 1500
        return eye_x, eye_y, eye_z, bx, by, bz + 2000
    elif cam_id == 2:
        eye_x = ox
        eye_y = oy + 500
        eye_z = oz - 1500
        return eye_x, eye_y, eye_z, ox, oy, oz + 2000
    elif cam_id == 3:
        eye_x = mid_x
        eye_y = 10000
        eye_z = mid_z
        return eye_x, eye_y, eye_z, mid_x, 0, mid_z
    elif cam_id == 4:
        eye_x = 0
        eye_y = 500
        eye_z = -FIELD_LENGTH / 2 - 1000
        return eye_x, eye_y, eye_z, bpx, 200, bpz
    else:
        eye_x = 0
        eye_y = 500
        eye_z = FIELD_LENGTH / 2 + 1000
        return eye_x, eye_y, eye_z, bpx, 200, bpz


def play(model_path="models/rl_best.pt", num_games=3, fps=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    hidden = state_dict["shared.0.weight"].shape[0]
    model = ActorCritic(OBS_SIZE, ACT_SIZE, hidden=hidden).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    env = make_env(team_size=1, tick_skip=32)

    pygame.init()
    pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("OrangeChicken RL Bot - 3D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16, bold=True)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [0, 1, 0, 0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.35, 0.35, 0.4, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.95, 1])
    glClearColor(0.04, 0.04, 0.08, 1.0)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(50, 1280 / 720, 50, 80000)
    glMatrixMode(GL_MODELVIEW)

    cam_id = 0

    obs_dict = env.reset()
    agent_ids = list(obs_dict.keys())
    blue_id = [a for a in agent_ids if "blue" in a][0]
    orange_id = [a for a in agent_ids if "orange" in a][0]
    obs = obs_dict[blue_id].flatten()

    games_done = 0
    total_goals = 0
    paused = False
    ball_trail = deque(maxlen=200)
    blue_trail = deque(maxlen=100)
    orange_trail = deque(maxlen=100)

    running = True
    while running and games_done < num_games:
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_q:
                    running = False
                elif K_1 <= event.key <= K_6:
                    cam_id = event.key - K_1

        if not paused:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value, std = model(obs_tensor)
                dist = torch.distributions.Normal(logits, std)
                action = dist.sample().squeeze(0).cpu().numpy()

            actions_dict = {a: action for a in agent_ids}
            next_obs_dict, rewards, terminated, truncated = env.step(actions_dict)
            done = terminated[blue_id] or truncated[blue_id]

            state = env.state
            bp = state.ball.position
            ball_trail.append((bp[0], bp[1]))

            cp = state.cars[blue_id].physics.position
            blue_trail.append((cp[0], cp[1]))

            op = state.cars[orange_id].physics.position
            orange_trail.append((op[0], op[1]))

            if done:
                if rewards[blue_id] > 5:
                    total_goals += 1
                games_done += 1
                if games_done < num_games:
                    obs_dict = env.reset()
                    agent_ids = list(obs_dict.keys())
                    blue_id = [a for a in agent_ids if "blue" in a][0]
                    orange_id = [a for a in agent_ids if "orange" in a][0]
                    ball_trail.clear()
                    blue_trail.clear()
                    orange_trail.clear()
                else:
                    obs_dict = next_obs_dict
                    obs = obs_dict[blue_id].flatten()
            else:
                obs_dict = next_obs_dict
                obs = obs_dict[blue_id].flatten()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        state = env.state
        bp = state.ball.position
        cp = state.cars[blue_id].physics.position
        op = state.cars[orange_id].physics.position

        ex, ey, ez, tx, ty, tz = get_camera(cam_id, cp, op, bp)
        glLoadIdentity()
        gluLookAt(ex, ey, ez, tx, ty, tz, 0, 1, 0)

        glDisable(GL_LIGHTING)
        draw_field()

        draw_sphere(bp[0], bp[2], bp[1], BALL_RADIUS, BALL_COLOR)

        blue_vel = state.cars[blue_id].physics.linear_velocity
        blue_yaw = math.atan2(blue_vel[0], blue_vel[1]) if np.linalg.norm(blue_vel) > 100 else 0
        draw_box(cp[0], cp[2] + 20, cp[1], 44, 40, 60, BLUE_COLOR, blue_yaw)
        draw_nose(cp[0], cp[2], cp[1], blue_yaw, BLUE_COLOR)

        orange_vel = state.cars[orange_id].physics.linear_velocity
        orange_yaw = math.atan2(orange_vel[0], orange_vel[1]) if np.linalg.norm(orange_vel) > 100 else 0
        draw_box(op[0], op[2] + 20, op[1], 44, 40, 60, ORANGE_COLOR, orange_yaw)
        draw_nose(op[0], op[2], op[1], orange_yaw, ORANGE_COLOR)

        draw_trail(list(ball_trail), (1, 1, 1))
        draw_trail(list(blue_trail), BLUE_COLOR)
        draw_trail(list(orange_trail), ORANGE_COLOR)

        glEnable(GL_LIGHTING)

        boost = state.cars[blue_id].boost_amount
        vel = np.linalg.norm(state.cars[blue_id].physics.linear_velocity)
        ball_vel = np.linalg.norm(state.ball.linear_velocity)

        render_hud(10, 10, f"Game {games_done + 1}/{num_games}  Goals: {total_goals}  Cam: [{CAM_NAMES[cam_id]}]", font)
        render_hud(10, 32, f"Boost: {boost:.0f}%  Speed: {vel:.0f}  Ball: {ball_vel:.0f}", font)
        render_hud(10, 54, "1-6: camera  SPACE: pause  Q: quit", font)

        if paused:
            render_hud(560, 340, "PAUSED", font, (255, 255, 0))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    print(f"\nResults: {games_done} games, {total_goals} goals")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rl_best.pt")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    play(model_path=args.model, num_games=args.games, fps=args.fps)
