import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from ai import Dqn

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

last_x = 0
last_y = 0
n_points = 0
length = 0

brain = Dqn(5, 3, 0.9)
action2rotation = [0, 20, -20]
last_reward = 0
scores = []

first_update = True


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global goals
    global goal_index
    goal_index = 0
    goals = []
    sand = np.zeros((longueur, largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False


last_distance = 0


class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x) - 10:int(self.sensor1_x) + 10,
                                  int(self.sensor1_y) - 10:int(self.sensor1_y) + 10])) / 400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x) - 10:int(self.sensor2_x) + 10,
                                  int(self.sensor2_y) - 10:int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x) - 10:int(self.sensor3_x) + 10,
                                  int(self.sensor3_y) - 10:int(self.sensor3_y) + 10])) / 400.
        if self.sensor1_x > longueur - 10 or self.sensor1_x < 10 or self.sensor1_y > largeur - 10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > longueur - 10 or self.sensor2_x < 10 or self.sensor2_y > largeur - 10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > longueur - 10 or self.sensor3_x < 10 or self.sensor3_y > largeur - 10 or self.sensor3_y < 10:
            self.signal3 = 1.


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global goals
        global goal_index

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.1
            if distance < last_distance:
                last_reward = (last_distance - distance) / 30.0

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 70:
            if len(goals) != 0:
                goal_index = (goal_index + 1) % len(goals)
                goal_x = goals[goal_index][0]
                goal_y = goals[goal_index][1]

        if action != 0:
            last_reward -= 0.1
        last_distance = distance


adding_point = False
erasing = False


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y, goals
        with self.canvas:
            if adding_point:
                Color(1, 1, 1)
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
                goals.append((touch.x, touch.y))
                print(goals)
                return
            if erasing:
                Color(0, 0, 0)
                sand[int(touch.x), int(touch.y)] = 0
            else:
                Color(0.8, 0.7, 0)
                sand[int(touch.x), int(touch.y)] = 1
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y, erasing
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = 0.5
            touch.ud['line'].width = int(20 * density + 1)
            if erasing:
                sand[int(touch.x) - 10: int(touch.x) + 10, int(touch.y) - 10: int(touch.y) + 10] = 0
            else:
                sand[int(touch.x) - 10: int(touch.x) + 10, int(touch.y) - 10: int(touch.y) + 10] = 1
            last_x = x
            last_y = y


class CarApp(App):
    point_btn = Button(text='check', size=(50, 50))
    erase_btn = Button(text='erase', size=(50, 50))

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clear_btn = Button(text='clear', size=(50, 50))
        save_btn = Button(text='save', pos=(parent.width / 2, 0), size=(50, 50))
        load_btn = Button(text='load', pos=(2 * parent.width / 2, 0), size=(50, 50))
        plot_btn = Button(text='plot', pos=(5 * parent.width / 2, 0), size=(50, 50))
        self.point_btn.pos = (3 * parent.width / 2, 0)
        self.erase_btn.pos = (2 * parent.width, 0)
        clear_btn.bind(on_release=self.clear_canvas)
        save_btn.bind(on_release=self.save)
        load_btn.bind(on_release=self.load)
        self.point_btn.bind(on_release=self.add_point)
        self.erase_btn.bind(on_release=self.erase)
        plot_btn.bind(on_release=self.plot)
        parent.add_widget(self.painter)
        parent.add_widget(clear_btn)
        parent.add_widget(save_btn)
        parent.add_widget(load_btn)
        parent.add_widget(self.point_btn)
        parent.add_widget(self.erase_btn)
        parent.add_widget(plot_btn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

    def add_point(self, obj):
        global adding_point
        if adding_point:
            self.point_btn.color = [1, 1, 1, 1]
        else:
            self.point_btn.color = [1, 0, 0, 1]
        adding_point = not adding_point

    def erase(self, obj):
        global erasing
        if erasing:
            self.erase_btn.color = [1, 1, 1, 1]
        else:
            self.erase_btn.color = [1, 0, 0, 1]
        erasing = not erasing

    def plot(self, obj):
        plt.plot(scores)
        plt.show()


if __name__ == '__main__':
    CarApp().run()
