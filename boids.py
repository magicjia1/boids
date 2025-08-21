import math
from random import randint, uniform

import pygame as pg
import pygame_gui

# Color constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Window Parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Parameters
NUM_BOIDS = 100
BOID_SIZE = 10
SPEED = 3
MAX_FORCE = 0.3
BOID_FRICTION = 0.75

WANDER_RADIUS = 30

SEPARATION = 2
SEPARATION_RADIUS = 40

ALIGNMENT = 1
ALIGNMENT_RADIUS = 50

COHESION = 1
COHESION_RADIUS = 80

# 在颜色常量后添加障碍物参数
OBSTACLE_RADIUS = 30  # 障碍物半径
OBSTACLE_AVOIDANCE = 80  # 避障力权重（可通过滑块调节）

import math
import pygame as pg
from pygame.math import Vector2

# 新增波浪障碍物参数
WAVE_AMPLITUDE = 50  # 波浪振幅（高度）
WAVE_LENGTH = 200    # 波浪长度（周期）
WAVE_SEGMENTS = 20   # 组成波浪的线段数量（越多数值越平滑）
WAVE_COLOR = (0, 255, 0)  # 波浪障碍物颜色（绿色）

class WaveObstacle:
    def __init__(self, start_pos, end_pos, amplitude=WAVE_AMPLITUDE, length=WAVE_LENGTH):
        """
        初始化波浪形障碍物
        :param start_pos: 波浪起点 (x, y)
        :param end_pos: 波浪终点 (x, y)
        :param amplitude: 波浪振幅（上下起伏的高度）
        :param length: 波浪周期长度
        """
        self.start = Vector2(start_pos)
        self.end = Vector2(end_pos)
        self.amplitude = amplitude
        self.wave_length = length  # 单个周期的长度
        self.segments = WAVE_SEGMENTS  # 用于绘制的线段数量
        self.points = self._generate_wave_points()  # 生成波浪的所有顶点

    def _generate_wave_points(self):
        """生成波浪形状的所有顶点（正弦曲线）"""
        points = []
        total_length = self.start.distance_to(self.end)  # 波浪总长度
        direction = (self.end - self.start).normalize()  # 波浪延伸方向
        perpendicular = Vector2(-direction.y, direction.x)  # 垂直于延伸方向（用于上下起伏）

        for i in range(self.segments + 1):
            # 沿波浪方向的进度（0到1）
            progress = i / self.segments
            # 计算当前点在直线方向上的位置
            line_pos = self.start + direction * (total_length * progress)
            # 计算正弦曲线的偏移量（上下起伏）
            wave_offset = math.sin(progress * (2 * math.pi) * (total_length / self.wave_length))
            # 叠加垂直方向的偏移，形成波浪
            wave_pos = line_pos + perpendicular * (self.amplitude * wave_offset)
            points.append(wave_pos)
        return points

    def draw(self, screen):
        """绘制波浪形障碍物（用多边形填充或线条）"""
        # 用绿色线条绘制波浪轮廓
        pg.draw.lines(screen, WAVE_COLOR, False, [ (p.x, p.y) for p in self.points ], 3)
        # 若需要填充（如实心障碍），可使用polygon，但需闭合形状
        # pg.draw.polygon(screen, WAVE_COLOR, [ (p.x, p.y) for p in self.points ])

class Obstacle:
    def __init__(self, position, radius=OBSTACLE_RADIUS):
        self.pos = pg.math.Vector2(position)  # 障碍物位置
        self.radius = radius  # 障碍物半径

    def draw(self, screen):
        # 绘制障碍物（红色圆形）
        pg.draw.circle(screen, (255, 0, 0), (int(self.pos.x), int(self.pos.y)), self.radius)

class Simulation:

    def __init__(self):
        pg.init()
        self.running = False
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen_rect = self.screen.get_rect()
        self.fps = 60

        # Set title of window
        pg.display.set_caption("Boids")

        # Load the icon image and set it as the window icon
        # icon = pg.image.load('boids.png')
        # pg.display.set_icon(icon)

        # Create boids
        self.boids = []
        for i in range(NUM_BOIDS):
            position = (randint(0, SCREEN_WIDTH), randint(0, SCREEN_HEIGHT))
            while any(boid.pos == position for boid in self.boids):
                position = (randint(0, SCREEN_WIDTH), randint(0, SCREEN_HEIGHT))

            self.boids.append(Boid(self, position))
        
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT), 'theme.json')

        self.separation_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((50, 10), (100, 25)),  # position for the first slider
            start_value=SEPARATION,
            value_range=(0, 5),
            manager=self.manager
        )
        self.alignment_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((200, 10), (100, 25)),  # position for the second slider
            start_value=ALIGNMENT,
            value_range=(0, 5),
            manager=self.manager
        )
        self.cohesion_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((350, 10), (100, 25)),  # position for the third slider
            start_value=COHESION,
            value_range=(0, 5),
            manager=self.manager
        )

        # Create UILabels for displaying the values
        self.separation_label = pygame_gui.elements.UILabel(
            relative_rect=pg.Rect((50, 40), (100, 20)),  # position below the first slider
            text=f"Separation: {SEPARATION}",
            manager=self.manager
        )
        self.alignment_label = pygame_gui.elements.UILabel(
            relative_rect=pg.Rect((200, 40), (100, 20)),  # position below the second slider
            text=f"Alignment: {ALIGNMENT}",
            manager=self.manager
        )
        self.cohesion_label = pygame_gui.elements.UILabel(
            relative_rect=pg.Rect((350, 40), (100, 20)),  # position below the third slider
            text=f"Cohesion: {COHESION}",
            manager=self.manager
        )
        # 添加障碍物（可自定义位置，这里随机生成3个）
        self.obstacles = [ # 水平波浪：从(100, 300)到(700, 300)
            WaveObstacle(start_pos=(100, 300), end_pos=(700, 300)),
            # 倾斜波浪：从(200, 150)到(600, 450)
            WaveObstacle(start_pos=(200, 150), end_pos=(600, 450), amplitude=30, length=150)]
        for _ in range(3):
            # 确保障碍物不超出屏幕
            pos = (
                randint(OBSTACLE_RADIUS, SCREEN_WIDTH - OBSTACLE_RADIUS*2),
                randint(OBSTACLE_RADIUS, SCREEN_HEIGHT - OBSTACLE_RADIUS*2)
            )
            self.obstacles.append(Obstacle(pos))

        # 添加避障力调节滑块（在原有滑块后）
        self.obstacle_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((500, 10), (100, 25)),
            start_value=OBSTACLE_AVOIDANCE,
            value_range=(5, 100),
            manager=self.manager
        )
        self.obstacle_label = pygame_gui.elements.UILabel(
            relative_rect=pg.Rect((500, 40), (100, 20)),
            text=f"Obstacle: {OBSTACLE_AVOIDANCE}",
            manager=self.manager
        )
        

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False

    def draw(self):
        # Empty the last screen
        self.screen.fill(BLACK)
        # 先绘制障碍物（避免被Boids遮挡）
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        # Draw all boids
        for boid in self.boids:
            boid.draw(self.screen)

        # Update the screen
        pg.display.update()

    def update(self):
        """
        Method for going one step in the simulation
        """
        for boid in self.boids:
            boid.update()

    def run(self):
        """
        Runs the simulation
        """
        self.running = True
        while self.running:
            self.clock.tick(self.fps)
            self.events()
            self.update()
            self.draw()

            time_delta = self.clock.tick(self.fps)/1000.0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

                self.manager.process_events(event)

            self.manager.update(time_delta)
            
            global SEPARATION, ALIGNMENT, COHESION  
            SEPARATION = self.separation_slider.get_current_value()
            ALIGNMENT = self.alignment_slider.get_current_value()
            COHESION = self.cohesion_slider.get_current_value()

            self.separation_label.set_text(f"Separation: {SEPARATION:.2f}")  # update text with current value
            self.alignment_label.set_text(f"Alignment: {ALIGNMENT:.2f}")    # update text with current value
            self.cohesion_label.set_text(f"Cohesion: {COHESION:.2f}")
            global OBSTACLE_AVOIDANCE
            OBSTACLE_AVOIDANCE = self.obstacle_slider.get_current_value()
            self.obstacle_label.set_text(f"Obstacle: {OBSTACLE_AVOIDANCE:.2f}")
            # Removed the line that was causing the error
            self.manager.draw_ui(self.screen)

            pg.display.update()

class PhysicsObjet:

    def __init__(self, simulation, position):
        self.simulation = simulation
        self.acc = pg.math.Vector2(0, 0)
        self.vel = pg.math.Vector2(0, 0)
        self.pos = pg.math.Vector2(position)

        self.speed = 1

        self.friction = 0.9

    def update(self):
        self.vel += self.acc
        self.pos += self.vel * self.speed

        # Reset acceleration
        self.acc *= 0

        # Simplistic surface friction
        self.vel *= self.friction

        # wrap around the edges of the screen
        if self.pos.x > self.simulation.screen_rect.w:
            self.pos.x -= self.simulation.screen_rect.w
        elif self.pos.x < 0:
            self.pos.x += self.simulation.screen_rect.w

        if self.pos.y > self.simulation.screen_rect.h:
            self.pos.y -= self.simulation.screen_rect.h
        elif self.pos.y < 0:
            self.pos.y += self.simulation.screen_rect.h


class Boid(PhysicsObjet):

    def __init__(self, simulation, position):
        super().__init__(simulation, position)
        self.speed = SPEED  # Max speed
        self.vel = pg.math.Vector2(randint(-2, 2), randint(-2, 2))  # Random initial velocity

        self.max_force = MAX_FORCE  # force cap, limits the size of the different forces
        self.friction = BOID_FRICTION  # Friction coefficient for the simplistic physics

        # Parameters for wandering behaviour
        self.target = pg.math.Vector2(0, 0)
        self.future_loc = pg.math.Vector2(0, 0)
        self.theta = uniform(-math.pi, math.pi)

    def update(self):
        """
        Updates the acceleration of the boid by adding together the different forces that acts on it
        """
        self.acc += self.wander()  # Wandering force
        self.acc += self.separation() * SEPARATION  # separation force scaled with a control parameter
        self.acc += self.alignment() * ALIGNMENT  # alignment force scaled with a control parameter
        self.acc += self.cohesion() * COHESION  # cohesion force scaled with a control parameter
        self.acc += self.avoid_obstacles() * OBSTACLE_AVOIDANCE  # 新增避障力
        # move by calling super
        super().update()

    def avoid_obstacles(self):
        """扩展避障逻辑，支持波浪形障碍物（线段集合）"""
        force_vector = Vector2(0, 0)
        safe_distance = BOID_SIZE * 3  # 安全距离（可调整）

        for obstacle in self.simulation.obstacles:
            if isinstance(obstacle, Obstacle):
                # 原有圆形障碍物的避障逻辑（保持不变）
                distance = self.pos.distance_to(obstacle.pos)
                if distance < (obstacle.radius + safe_distance) and distance > 0:
                    away_dir = (self.pos - obstacle.pos).normalize()
                    force_strength = (obstacle.radius + safe_distance - distance) / (obstacle.radius + safe_distance)
                    force_vector += away_dir * force_strength * self.max_force

            elif isinstance(obstacle, WaveObstacle):
                # 波浪形障碍物（线段集合）的避障逻辑
                min_distance = float('inf')
                closest_dir = Vector2(0, 0)

                # 遍历波浪的每一段线段（相邻两点组成线段）
                for i in range(len(obstacle.points) - 1):
                    p1 = obstacle.points[i]
                    p2 = obstacle.points[i + 1]
                    # 计算Boid到线段p1-p2的最短距离和方向
                    dist, dir_vec = self._distance_to_segment(p1, p2)
                    if dist < min_distance:
                        min_distance = dist
                        closest_dir = dir_vec

                # 若距离小于安全距离，生成避障力
                if min_distance < safe_distance and min_distance > 0:
                    # 方向：远离线段，力的大小与距离成反比
                    force_strength = (safe_distance - min_distance) / safe_distance
                    force_vector += closest_dir * force_strength * self.max_force

        return force_vector

    def _distance_to_segment(self, p1, p2):
            """计算点（self.pos）到线段p1-p2的最短距离和方向向量"""
            # 线段向量
            seg_vec = p2 - p1
            # 点到线段起点的向量
            point_vec = self.pos - p1
            # 计算投影比例（0~1表示点在投影在线段上）
            proj = max(0, min(1, point_vec.dot(seg_vec) / seg_vec.length_squared()))
            # 投影点
            closest = p1 + seg_vec * proj
            # 距离和方向向量（从线段指向点）
            distance = self.pos.distance_to(closest)
            direction = (self.pos - closest).normalize() if distance > 0 else Vector2(0, 0)
            return distance, direction

    def avoid_obstacles(self):
            """扩展避障逻辑，支持波浪形障碍物（线段集合）"""
            force_vector = Vector2(0, 0)
            safe_distance = BOID_SIZE * 3  # 安全距离（可调整）

            for obstacle in self.simulation.obstacles:
                if isinstance(obstacle, Obstacle):
                    # 原有圆形障碍物的避障逻辑（保持不变）
                    distance = self.pos.distance_to(obstacle.pos)
                    if distance < (obstacle.radius + safe_distance) and distance > 0:
                        away_dir = (self.pos - obstacle.pos).normalize()
                        force_strength = (obstacle.radius + safe_distance - distance) / (
                                    obstacle.radius + safe_distance)
                        force_vector += away_dir * force_strength * self.max_force

                elif isinstance(obstacle, WaveObstacle):
                    # 波浪形障碍物（线段集合）的避障逻辑
                    min_distance = float('inf')
                    closest_dir = Vector2(0, 0)

                    # 遍历波浪的每一段线段（相邻两点组成线段）
                    for i in range(len(obstacle.points) - 1):
                        p1 = obstacle.points[i]
                        p2 = obstacle.points[i + 1]
                        # 计算Boid到线段p1-p2的最短距离和方向
                        dist, dir_vec = self._distance_to_segment(p1, p2)
                        if dist < min_distance:
                            min_distance = dist
                            closest_dir = dir_vec

                    # 若距离小于安全距离，生成避障力
                    if min_distance < safe_distance and min_distance > 0:
                        # 方向：远离线段，力的大小与距离成反比
                        force_strength = (safe_distance - min_distance) / safe_distance
                        force_vector += closest_dir * force_strength * self.max_force

            return force_vector



    def separation(self):
        """
        Calculate the separation force vector
        Separation: steer to avoid crowding local flockmates
        :return force vector
        """
        force_vector = pg.math.Vector2(0, 0)
        boids_in_view = self.boids_in_radius(SEPARATION_RADIUS)
    
        # Early return if there are no boids in radius
        if len(boids_in_view) == 0:
            return force_vector
        
        # TODO: Implement this
        for other_boid in boids_in_view:
            distance = self.pos.distance_to(other_boid.pos)

            # Make sure the distance is not 0 to avoid division by 0
            if distance == 0:
                continue

            # Calculate the force vector, the closer the boid is the larger the force
            x_diff = self.pos.x - other_boid.pos.x
            y_diff = self.pos.y - other_boid.pos.y
            force_vector += pg.math.Vector2(x_diff, y_diff) * (SEPARATION_RADIUS / distance)
        
        force_vector = self.cap_force(force_vector, boids_in_view)
        return force_vector

    def alignment(self):
        """
        Calculate the alignment force vector
        Alignment: steer towards the average heading of local flockmates
        :return force vector
        """
        force_vector = pg.math.Vector2(0, 0)
        boids_in_view = self.boids_in_radius(ALIGNMENT_RADIUS)
        
        # Early return if there are no boids in radius
        if len(boids_in_view) == 0:
            return force_vector
        
        # Find the direction of the flock by adding together the velocity vectors of the boids in view
        for other_boid in boids_in_view:
            force_vector += other_boid.vel
            
        if force_vector.length() == 0:
            return force_vector
        
        force_vector = self.cap_force(force_vector, boids_in_view)
        return force_vector

    def cohesion(self):
        """
        Calculate the cohesion force vector
        Cohesion: steer to move toward the average position of local flockmates
        """
        force_vector = pg.math.Vector2(0, 0)
        boids_in_view = self.boids_in_radius(COHESION_RADIUS)
        
        # Early return if there are no boids in radius
        if len(boids_in_view) == 0:
            return force_vector
        
        # Calculate the average position of the boids in view
        other_boid: Boid
        for other_boid in boids_in_view:
            # Make the boids move towards the average position of the boids in view
            dx = other_boid.pos.x - self.pos.x
            dy = other_boid.pos.y - self.pos.y
            force_vector += pg.math.Vector2(dx, dy)

        force_vector = self.cap_force(force_vector, boids_in_view)
        return force_vector

    
    def boids_in_radius(self, radius: float) -> list:
        """
        Find all boids in a given radius
        """
        boids: list = []
        for other_boid in self.simulation.boids:
            if other_boid == self:
                continue

            if self.pos.distance_to(other_boid.pos) < radius:
                boids.append(other_boid)
        return boids

    def cap_force(self, force_vector: pg.math.Vector2, boids_in_view: list) -> pg.math.Vector2:
        """
        Takes a list of boids in view and returns a force vector that is capped by the max force
        """
        force_vector /= len(boids_in_view)
        # Make sure the force vector is not 0
        if force_vector.length() <= 0:
            return force_vector
        
        force_vector = force_vector.normalize() * self.speed - self.vel

        if force_vector.length() > self.max_force:
            force_vector.scale_to_length(self.max_force)
        return force_vector

    def move_towards_target(self, target):
        """
        Calculate force vector for moving the boid to the target
        """
        # vector to the target
        desired = target - self.pos

        distance = desired.length()
        desired = desired.normalize()

        # Radius
        radius = 100

        if distance < radius:
            # if the distance is less than the radius,
            m = remap(distance, 0, radius, 0, self.speed)

            # scale the desired vector up to continue movement in that direction
            desired *= m
        else:
            desired *= self.speed

        force_vector = desired - self.vel
        limit(force_vector, self.max_force)
        return force_vector

    def wander(self):
        """
        Calcualte a random target to move towards to get natural random flight
        """
        if self.vel.length_squared() != 0:
            # Calculate where you will be in the future
            self.future_loc = self.vel.normalize() * 80

            # Calculate a random angle addition
            self.theta += uniform(-math.pi, math.pi) / 10

            # set the target to your position + your future position + a distance in the direction of the random angle
            self.target = self.pos + self.future_loc + pg.math.Vector2(WANDER_RADIUS * math.cos(self.theta),
                                                                       WANDER_RADIUS * math.sin(self.theta))
        return self.move_towards_target(self.target)

    def draw(self, screen):
        """Draw boid to screen"""

        # Calculate the angle to the velocity vector to get the forward direction
        angle = math.atan2(self.vel.y, self.vel.x)
        other_points_angle = 0.75 * math.pi  # angle +- value to get the other two points in the triangle

        # Get the points of the triangle
        x0 = self.pos.x + BOID_SIZE * math.cos(angle)
        y0 = self.pos.y + BOID_SIZE * math.sin(angle)

        x1 = self.pos.x + BOID_SIZE * math.cos(angle + other_points_angle)
        y1 = self.pos.y + BOID_SIZE * math.sin(angle + other_points_angle)

        x2 = self.pos.x + BOID_SIZE * math.cos(angle - other_points_angle)
        y2 = self.pos.y + BOID_SIZE * math.sin(angle - other_points_angle)

        # Draw
        pg.draw.polygon(screen, WHITE, [(x1, y1), (x2, y2), (x0, y0)])


# Helper functions
def remap(n, start1, stop1, start2, stop2):
    """Remap a value in one range to a different range"""
    new_value = (n - start1) / (stop1 - start1) * (stop2 - start2) + start2
    if start2 < stop2:
        return constrain(new_value, start2, stop2)
    else:
        return constrain(new_value, stop2, start2)


def constrain(n, low, high):
    """Constrain a value to a range"""
    return max(min(n, high), low)


def limit(vector, length):
    """Cap a value"""
    if vector.length_squared() <= length * length:
        return
    else:
        vector.scale_to_length(length)


if __name__ == '__main__':
    sim = Simulation()
    sim.run()
