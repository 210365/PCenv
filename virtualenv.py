import gym
from gym import spaces
import numpy as np
import pygame
import random
import math
from scipy.stats import poisson
from collections import namedtuple, deque
import cv2
from skimage.measure import label,regionprops
import pygame.surfarray
import threading
from PIL import Image,PngImagePlugin
from PIL.PngImagePlugin import PngInfo
import os
from gym.envs.mine import SDIjisuan
import time
import pickle
from datetime import datetime
import torch
from torchvision.transforms import Resize
import threading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


#分阶段修改以下参数，实现不同流速场
liusu=100
WINDOW_WIDTH=1600
WINDOW_HEIGHT = 800
GRID_SIZE = 100
GRID_SIZE_W=100
GRID_SIZE_H=200
suiji_EFFECT=20
FLOW_SPEED_SCALE = 0.01
a=100
w=0.003
p=500
k=300

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = (255, 255, 255)
        self.radius = 1

    def move(self, flow_map,i):
        grid_x = min(max(int(self.x ), 0), flow_map[i].shape[0] - 1)
        grid_y = min(max(int(self.y ), 0), flow_map[i].shape[1] - 1)
        flow = flow_map[i][grid_x][grid_y]
        self.x += flow[0] * FLOW_SPEED_SCALE
        self.y += flow[1] * FLOW_SPEED_SCALE

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (int(self.x), int(self.y),2,2))



class NewDroneParticlesEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        super(NewDroneParticlesEnv, self).__init__()

        # self.tarandrea_area = deque([], maxlen=2)

        self.grid_width = WINDOW_WIDTH // GRID_SIZE
        self.grid_height = WINDOW_HEIGHT // GRID_SIZE

        self.action_space = spaces.Discrete(32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        self.particles = []
        self.drone_pos = [WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2]  # 无人机初始位置
        self.blink_timer = 0

        self.flow_map = self.generate_flow_map(liusu)

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.wind_num=3000
        self.random1_wind=0
        self.random2_wind=0
        self.pianyi_x=0
        self.pianyi_y=0
        self.mySDI=0
        self.flow_map_count=0
        self.done_count=0

        self.random3_wind = 0
        self.random4_wind = 0
    def quadratic_curve(self,x):
        # a=np.arange(100,300,20)
        return a * math.sin(w * x + p) + k, a * w * math.cos(w * x + p)

    def quadratic_curve2(self,x,a,w,p,k):
        return a * math.sin(w * x + p) + k, a * w * math.cos(w * x + p)
    def generate_flow_map(self,a):
        sigma = 500
        flow_map = np.zeros((1,WINDOW_WIDTH, WINDOW_HEIGHT, 2), dtype=np.float64)

        for num in range(flow_map.shape[0]):


            for i in range(flow_map.shape[1]):
                for j in range(flow_map.shape[2]):


                    distance_from_curves1 = abs(i - 800)


                    speed1 = 40 * (math.exp(-(distance_from_curves1) ** 2 / (2 * 1000 ** 2)))

                    speed_y = -speed1
                    speed_x = random.uniform(-speed_y*0.01,speed_y*0.01)
                    flow_map[num][i][j] = [ 0.1*speed_x,  speed_y]  # update
            np.save('E:/flowmap/flow_map.npy', flow_map)
        return flow_map


    def ROI_SELECT(self):
        ROI=[]
        for i in range(3):
            roi_sdi = SDIjisuan.main_function_seeding_metrics('E:/observationframe/', [0, 0], 'png', 0.7,
                                                                           (0, 200+i*100, 1600, 400), 50, 0)
            ROI.append(roi_sdi[0])
        min_roi=min(ROI)
        index=ROI.index(min_roi)
        return index

    def rewardjisuan_2(self, pianyi_xy,action,obv,hight):
        roi_sdi = SDIjisuan.main_function_seeding_metrics('E:/observationframe/', [0, 0], 'png', 0.7,(0, 200+hight*100, 1600, 400), 50, 0)
        self.mySDI=roi_sdi[0]

        sdi_at=obv[action]

        reward = -self.mySDI - 1 / sdi_at-0.02*pianyi_xy

        return reward


    def windopen(self):
        # 分阶段修改以下参数，实现不同风力
        low=50
        high=70
        random1=np.random.uniform(low,high)
        random2=np.random.uniform(random1-10,random1+10)

        t=max(random1,random2)
        random1=min(random2,random1)
        random2=t


        return random1,random2

    def split_image(self,input_folder, output_folder,hight):


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        for file_name in os.listdir(input_folder):

            file_path = os.path.join(input_folder, file_name)


            if file_name.lower().endswith('.png'):
                try:

                    with Image.open(file_path) as img:

                        if img.size == (1600, 800):

                            count = 1
                            for row in range(2):  # 5行
                                for col in range(16):  # 8列

                                    left = col * 100
                                    upper = row * 200+200+hight*100
                                    right = left + 100
                                    lower = upper + 200


                                    cropped_img = img.crop((left, upper, right, lower))

                                    # 保存小图，命名为0001, 0002, ...
                                    output_file_name = f"{count:04d}.png"
                                    output_path = os.path.join(output_folder, output_file_name)
                                    cropped_img.save(output_path)

                                    count += 1
                            # print(f"处理完成: {file_name}")
                        else:
                            print(f"跳过文件（尺寸不匹配）: {file_name}")
                except Exception as e:
                    print(f"处理文件时出错: {file_name}, 错误: {e}")

    def get_screen(self):
        resize = T.Compose([T.ToPILImage(),
                            T.Grayscale(num_output_channels=1),
                            T.Resize((84, 84), interpolation=InterpolationMode.BICUBIC),
                            T.ToTensor()])

        # Transpose it into torch order (CHW).
        screen = self.render(mode='rgb_array').transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0)

    def get_conbined(self,extra):
        image=self.get_screen()
        image = image.view(image.size(0), -1)
        # print(image.shape)
        extra=torch.tensor(extra,dtype=torch.float32)
        # extra = extra.view(extra.size(0),-1)
        # print(extra.shape)
        extra = extra.unsqueeze(0)
        # print(extra.shape)
        combined_input=torch.cat((image,extra),dim=1)

        return combined_input




    def step(self,action):

        done = False
        hight=self.ROI_SELECT()
        grid_x=action % 16
        grid_y=action // 16

        x=grid_x*GRID_SIZE_W + GRID_SIZE_W//2
        y=grid_y*GRID_SIZE_H + GRID_SIZE_H//2 +200+hight*100

        num_particles = np.random.poisson(20, 1)[0]
        if (self.wind_num==3000):
            self.random1_wind, self.random2_wind = self.windopen()

            self.random3_wind = np.random.uniform(-self.random1_wind, self.random2_wind)
            self.random4_wind = np.random.uniform(-self.random1_wind, self.random2_wind)

        for _ in range(20):
            x_offset = np.random.normal(0, suiji_EFFECT)
            y_offset = np.random.normal(0, suiji_EFFECT)



            self.pianyi_x = self.random3_wind
            self.pianyi_y = self.random4_wind
            #50 40 30
            self.random3_wind=0
            self.random4_wind = -50
            particle = Particle(x + x_offset + self.random3_wind, y + y_offset + self.random4_wind)
            self.particles.append(particle)
        # pianyi_x=self.pianyi_x//20
        # pianyi_y=self.pianyi_y//20
        self.pianyi_x = (self.pianyi_x)
        self.pianyi_y = (self.pianyi_y)
        self.render(mode='rgb_array')
        # self.wind_num -= 1
        # if self.wind_num==0:
        #     self.wind_num=10000
        pygame.image.save(self.screen, 'E:/observationframe/0001.png')
        pianyi = math.sqrt(self.random3_wind**2 + self.random4_wind**2)
        print('____________')
        print(pianyi)
        print(self.random3_wind)
        print(self.wind_num)
        # kangxing_pianyi=pianyi/pianyi

        self.split_image('E:/observationframe/', 'E:/gridframe/',hight)
        observation, seeddensity = SDIjisuan.main_function_seeding_metrics2('E:/gridframe/', [0, 31], 'png', 0.7,
                                                                            (0, 0, 100, 200), 50, 0)
        # print(observation)
        reward = self.rewardjisuan_2(pianyi, action,observation,hight)
        self.done_count += 1
        print(self.done_count)
        print('____________')
        # print(self.mySDI)
        self.wind_num -= 1
        if self.wind_num == 0:
            self.wind_num = 3000
        if self.mySDI < 1:
            done = True
            self.done_count=0
            reward=100
            print('**********************')

        else:
            done = False
        if done ==True:
            self.wind_num -= 1
            if self.wind_num == 0:
                self.wind_num = 100

        if self.done_count == 100:
            self.done_count = 0
            if self.mySDI < 0.9:

                reward = 100



            done = True
        info = {}


        return _, reward, done, info








    def reset(self):
        self.particles = []
        self.drone_pos = [WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2]

        num_particles = np.random.poisson(200, 1)[0]

        for j in range(3):
            for i in range(num_particles):
                y_lizi = 700 - j * 100
                x_lizi = (WINDOW_WIDTH / 16) + j * 600+100
                x_offset = np.random.normal(0, 60)
                y_offset = np.random.normal(0, 60)
                particle = Particle(x_lizi + x_offset, y_lizi + y_offset)
                self.particles.append(particle)
        for i  in range(30):
            x=np.random.uniform(0,1600)
            y=np.random.uniform(200,800)
            num_particles = np.random.poisson(20, 1)[0]
            for _ in range(num_particles):
                x_offset = np.random.normal(0, suiji_EFFECT)
                y_offset = np.random.normal(0, suiji_EFFECT)


                particle = Particle(x + x_offset, y + y_offset+50 )
                self.particles.append(particle)

        return
    def update_particles(self):
        self.flow_map_count=self.done_count // 500
        for particle in self.particles:
            particle.move(self.flow_map,self.flow_map_count)
        self.particles = [p for p in self.particles if 0 <= p.x <= WINDOW_WIDTH and 0 <= p.y <= WINDOW_HEIGHT]

    def capture_and_save_frame(screen, frame_num, folder_num, parent_folder='E:/captured_images/'):

        subfolder_name = f"{folder_num:04d}"
        subfolder_path = os.path.join(parent_folder, subfolder_name)


        frame_image = screen.copy()


        image_name = f"{frame_num:04d}.png"
        image_path = os.path.join(subfolder_path, image_name)
        pygame.image.save(frame_image, image_path)


    def create_folders(total_loops, parent_folder='E:/captured_images/'):

        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)


        for folder_num in range(1, total_loops + 1):
            subfolder_name = f"{folder_num:04d}"
            subfolder_path = os.path.join(parent_folder, subfolder_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)


    def capture_images(total_loops=1000, frames_per_loop=30, parent_folder='E:/captured_images/'):

        create_folders(total_loops, parent_folder)

        frame_count = 0
        clock = pygame.time.Clock()


        for folder_num in range(1, total_loops + 1):
            for frame_num in range(frames_per_loop):

                screen = pygame.display.get_surface()


                if screen is None:
                    print("Error: No pygame screen surface available!")
                    return


                capture_and_save_frame(screen, frame_count + frame_num, folder_num, parent_folder)

                frame_count += 1

                clock.tick(30)



    def render(self,mode):

        self.screen.fill((0,0,0))

        roi_rect = pygame.Rect(0, 0, 1600, 200)
        border_color = (255, 0, 0)
        border_thickness = 1


        self.update_particles()
        for particle in self.particles:
            particle.draw(self.screen)
        self.update_particles()
        pygame.display.flip()

        if mode == "rgb_array":
            screen_array = pygame.surfarray.array3d(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            return screen_array

        elif mode == "human":
            captured_image = self.screen.subsurface(roi_rect).copy()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            return captured_image


























