import numpy as np
import cv2
import os
from skimage import measure
from matplotlib import pyplot as plt
# from gym.envs.mine.newdongtai import DroneParticleEnv
import os
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
# 全局变量
h1 = None

# 主函数，计算Seeding Metrics
def main_function_seeding_metrics(filefolder,contIniFin, ImFormat, BinThreshold, ROI, blockSizeR, MinValueAreaTracers):
    global h1
    wbar = 0
    contwaitbar = 0

    # 获取图像文件列表
    direc = sorted(os.listdir(filefolder))
    filenames =[file for file in direc if file.endswith(ImFormat)]
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
    # print(filenames)
    # 初始化变量
    num_frames = 600 # 将帧数固定为12
    MeanAreaFiltered = np.full((num_frames), np.nan)
    SeedingDensity = np.copy(MeanAreaFiltered)
    CV_Area = np.copy(MeanAreaFiltered)
    Dispersion = np.copy(MeanAreaFiltered)

    num_images = len(filenames)
    for cont in range( num_images):  # 确保不会超出文件数量
        contwaitbar += 1
        I = cv2.imread(f'{filefolder}/{filenames[cont]}', cv2.IMREAD_GRAYSCALE)
        m,n=I.shape
        # print(cont)
        # print(m,n)
        I = I.astype(np.uint8)

        # 二值化
        _, BW = cv2.threshold(I, int(BinThreshold * 255), 255, cv2.THRESH_BINARY)


        BW2 = np.ones_like(BW)
        BW = BW2 * BW
        BW = BW[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]  # 裁剪ROI

        # MSER特征检测
        mser = cv2.MSER_create()
        mser.setMinArea(0)
        mser.setMaxArea(14000)
        # 计算连通区域的标签
        labeled_image = measure.label(BW)

        # 计算区域属性
        regions = measure.regionprops(labeled_image)

        # 计算每个轮廓的面积
        Area = [region.area for region in regions]
        # print(Area)
        # 过滤掉异常的区域
        AreaFiltered = np.array(Area)
        AreaFiltered[AreaFiltered < 4] = np.nan
        AreaFiltered[AreaFiltered > (blockSizeR / 2) * (blockSizeR / 2)] = np.nan
        # print(cont)
        # print(num_images)
        MeanAreaFiltered[cont] = np.nanmean(AreaFiltered)
        # print(MeanAreaFiltered[cont])

        sAreaFiltered = np.ceil(AreaFiltered / MeanAreaFiltered[cont])
        sAreaFiltered = sAreaFiltered[~np.isnan(sAreaFiltered)]
        NumberParticles = np.sum(sAreaFiltered)
        # print(cont)
        # print(NumberParticles)
        #
        # print(AreaFiltered)
        # print(MeanAreaFiltered[cont])
        # print(sAreaFiltered)

        # 计算SeedingDensity和CV_Area
        SeedingDensity[cont] = NumberParticles / (ROI[3] * ROI[2])

        CV_Area[cont] = np.sqrt(np.nanvar(AreaFiltered)) / MeanAreaFiltered[cont]
        Dispersion[cont] = aggregation_function(BW, blockSizeR, blockSizeR, MinValueAreaTracers, MeanAreaFiltered[cont])

        # print(SeedingDensity[cont])
        # print(Dispersion[cont])
    # 计算平均值和SDI
    MeanDensity = np.nanmean(SeedingDensity)

    MeanCV_Area = np.nanmean(CV_Area)

    MeanNu = np.nanmean(Dispersion)
    MeanNu = np.nan_to_num(MeanNu, nan=0)
    Dispersion = np.nan_to_num(Dispersion, nan=0)
    CV_Area = np.nan_to_num(CV_Area, nan=0)
    SDI = ((Dispersion ** 0.1) / (SeedingDensity / 1.52E-03))
    SDI = np.nan_to_num(SDI, nan=50 )
    SeedingDensity = np.nan_to_num(SeedingDensity, nan=0.001)
    # print('**********************')
    # print(SDI)
    # print('**********************')
    # 绘制结果
    # plot_seeding_metrics(SeedingDensity, Dispersion, CV_Area, SDI)
    # print('222222222')
    # print(SDI)
    return  SDI


def main_function_seeding_metrics2(filefolder,contIniFin, ImFormat, BinThreshold, ROI, blockSizeR, MinValueAreaTracers):
    global h1
    wbar = 0
    contwaitbar = 0

    # 获取图像文件列表
    direc = sorted(os.listdir(filefolder))
    filenames =[file for file in direc if file.endswith(ImFormat)]
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
    # print(filenames)
    # 初始化变量
    num_frames = 32  # 将帧数固定为12
    MeanAreaFiltered = np.full((num_frames), np.nan)
    SeedingDensity = np.copy(MeanAreaFiltered)
    CV_Area = np.copy(MeanAreaFiltered)
    Dispersion = np.copy(MeanAreaFiltered)

    num_images = len(filenames)
    for cont in range( num_images):  # 确保不会超出文件数量
        contwaitbar += 1
        I = cv2.imread(f'{filefolder}/{filenames[cont]}', cv2.IMREAD_GRAYSCALE)
        m,n=I.shape
        # print(cont)
        # print(m,n)
        I = I.astype(np.uint8)

        # 二值化
        _, BW = cv2.threshold(I, int(BinThreshold * 255), 255, cv2.THRESH_BINARY)


        BW2 = np.ones_like(BW)
        BW = BW2 * BW
        BW = BW[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]  # 裁剪ROI

        # MSER特征检测
        mser = cv2.MSER_create()
        mser.setMinArea(0)
        mser.setMaxArea(14000)
        # 计算连通区域的标签
        labeled_image = measure.label(BW)

        # 计算区域属性
        regions = measure.regionprops(labeled_image)

        # 计算每个轮廓的面积
        Area = [region.area for region in regions]
        # print(Area)
        # 过滤掉异常的区域
        AreaFiltered = np.array(Area)
        AreaFiltered[AreaFiltered < 4] = np.nan
        AreaFiltered[AreaFiltered > (blockSizeR / 2) * (blockSizeR / 2)] = np.nan
        # print(cont)
        # print(num_images)
        MeanAreaFiltered[cont] = np.nanmean(AreaFiltered)
        # print(MeanAreaFiltered[cont])

        sAreaFiltered = np.ceil(AreaFiltered / MeanAreaFiltered[cont])
        sAreaFiltered = sAreaFiltered[~np.isnan(sAreaFiltered)]
        NumberParticles = np.sum(sAreaFiltered)
        # print(cont)
        # print(NumberParticles)
        #
        # print(AreaFiltered)
        # print(MeanAreaFiltered[cont])
        # print(sAreaFiltered)

        # 计算SeedingDensity和CV_Area
        SeedingDensity[cont] = NumberParticles / (ROI[3] * ROI[2])

        CV_Area[cont] = np.sqrt(np.nanvar(AreaFiltered)) / MeanAreaFiltered[cont]
        Dispersion[cont] = aggregation_function(BW, blockSizeR, blockSizeR, MinValueAreaTracers, MeanAreaFiltered[cont])

        # print(SeedingDensity[cont])
        # print(Dispersion[cont])
    # 计算平均值和SDI
    MeanDensity = np.nanmean(SeedingDensity)

    MeanCV_Area = np.nanmean(CV_Area)

    MeanNu = np.nanmean(Dispersion)
    MeanNu = np.nan_to_num(MeanNu, nan=0)
    Dispersion = np.nan_to_num(Dispersion, nan=0)
    CV_Area = np.nan_to_num(CV_Area, nan=0)
    SDI = ((Dispersion ** 0.1) / (SeedingDensity / 1.52E-03))
    SDI = np.nan_to_num(SDI, nan=50 )
    SeedingDensity = np.nan_to_num(SeedingDensity, nan=0.001)
    # print('**********************')
    # print(SDI)
    # print('**********************')
    # 绘制结果
    # plot_seeding_metrics(SeedingDensity, Dispersion, CV_Area, SDI)
    # print('222222222')
    # print(SDI)
    return  SDI,SeedingDensity
# 计算分散度的函数
def aggregation_function(Frame, blockSizeR, blockSizeC, MinValueAreaTracers, MeanAreaFiltered):
    rows, columns = Frame.shape
    blockVectorR = [blockSizeR] * (rows // blockSizeR) + [rows % blockSizeR]
    blockVectorC = [blockSizeC] * (columns // blockSizeC) + [columns % blockSizeC]

    NumberParticles = []

    # 将图像划分成块并分析每一块
    for r_start in range(0, rows, blockSizeR):
        for c_start in range(0, columns, blockSizeC):
            block = Frame[r_start:r_start + blockSizeR, c_start:c_start + blockSizeC]

            mser = cv2.MSER_create()
            mser.setMinArea(1)
            mser.setMaxArea(14000)
            # 计算连通区域的标签
            labeled_image = measure.label(block)

            regions = measure.regionprops(labeled_image)
            Area = [region.area for region in regions]

            AreaFiltered = np.array(Area)
            AreaFiltered[AreaFiltered < MinValueAreaTracers] = np.nan
            AreaFiltered[AreaFiltered > (blockSizeR / 2) * (blockSizeR / 2)] = np.nan
            sAreaFiltered = np.ceil(AreaFiltered / MeanAreaFiltered)
            sAreaFiltered = sAreaFiltered[~np.isnan(sAreaFiltered)]
            NumberParticles.append(np.sum(sAreaFiltered))

    # 计算分散度（D*）
    Dispersion = np.nanvar(NumberParticles) / np.nanmean(NumberParticles)
    return Dispersion


# 绘图函数
def plot_seeding_metrics(SeedingDensity, Dispersion, CV_Area, SDI):
    frame_numbers = np.arange(1, 13)  # 设置Frame Number范围为1到12
    plt.figure(figsize=(8, 6))

    plt.subplot(4, 1, 1)
    plt.plot(frame_numbers, SeedingDensity)
    plt.xlabel('Frame Number')
    plt.ylabel('ρ (ppp)')

    plt.subplot(4, 1, 2)
    plt.plot(frame_numbers, Dispersion)
    plt.xlabel('Frame Number')
    plt.ylabel('D*')

    plt.subplot(4, 1, 3)
    plt.plot(frame_numbers, CV_Area)
    plt.xlabel('Frame Number')
    plt.ylabel('CV Area')

    plt.subplot(4, 1, 4)
    plt.plot(frame_numbers, SDI)
    plt.xlabel('Frame Number')
    plt.ylabel('SDI')

    plt.tight_layout()
    plt.show()


def process_images(timestamps, image_folder, output_folder):

    def convert_to_timestamp(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp()

    timestamps = [convert_to_timestamp(ts) for ts in timestamps]
    print(f"Loaded timestamps: {timestamps}")
    print(f"Number of timestamps: {len(timestamps)}")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载所有图像文件名，确保按照文件名（时间戳）排序
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    print(f"Total image files found: {len(image_files)}")

    for idx, timestamp in enumerate(timestamps):
        print(f"Processing timestamp: {timestamp}")
        nearest_images = []

        # 寻找与当前时间戳最接近的两张图像
        for i, image_file in enumerate(image_files):
            try:
                img_with_metadata = Image.open(f'E:/frame/{image_file}')
                image_time = img_with_metadata.info["Time"]
                image_time = convert_to_timestamp(image_time)
            except ValueError:
                print(f"Skipping file with invalid name format: {image_file}")
                continue

            if image_time-1 <= timestamp<=image_time+1:
                nearest_images.append(i)

        # if len(nearest_images) < 2:
        #     print(f"Not enough images found for timestamp {timestamp}. Skipping...")
        #     continue

        # 创建固定格式的时间戳文件夹，比如 "timestamp_0001"
        timestamp_folder = os.path.join(output_folder, f'timestamp_{idx+1:04d}')
        print(f"Creating folder: {timestamp_folder}")
        os.makedirs(timestamp_folder, exist_ok=True)

        # 处理找到的最近两张图像，确保每次处理不同的图像
        for i, image_idx in enumerate(nearest_images):  # 处理相邻的两张图像
            img_path = os.path.join(image_folder, image_files[image_idx])
            print(f"Processing image: {img_path}")

            # 加载图像并转换为灰度图
            img = Image.open(img_path).convert('L')

            width, height = img.size
            img_counter = 1  # 用于给小图像编号
            for y in range(0, height, 200):
                for x in range(0, width, 200):
                    # 分割200x200的小图像
                    box = (x, y, min(x + 200, width), min(y + 200, height))
                    small_img = img.crop(box)

                    # 为每张图像创建单独的文件夹
                    subfolder = os.path.join(timestamp_folder, f'image_{i+1}')
                    os.makedirs(subfolder, exist_ok=True)

                    # 将小图像保存为 1.png, 2.png,...12.png
                    small_img_filename = os.path.join(subfolder, f'{img_counter}.png')
                    small_img.save(small_img_filename)
                    img_counter += 1  # 递增图像编号

    print("处理完成！")

def printdata_png():
    img_with_metadata = Image.open('E:/frame/000300.png')

    # 检查元数据并提取时间信息
    if "Time" in img_with_metadata.info:
        extracted_time = img_with_metadata.info["Time"]
        extracted_datetime = img_with_metadata.info["DateTime"]
        print("提取的时间信息:", extracted_time)
        print("提取的日期时间信息:", extracted_datetime)
    else:
        print("没有找到时间信息。")

def SCR_go():
    waifolder='E:/frameout/'
    target=[]
    targetnum=0
    gama=1.2
    scr=[]
    lidu=8
    SCR_JIHE=np.zeros((20,40,3,4))
    SDI_JIHE=np.zeros((20,40,3,4))
    SD_JIHE=np.zeros((20,40,3,4))
    SCR_ACTION=[]
    # 获取文件夹下的所有条目
    entries = os.listdir(waifolder)
    # 过滤出子文件夹
    subdirectories_1 = sorted_files = sorted(entries, key=lambda x: int(x.split('_')[1]))
    layer_1_num=len(subdirectories_1)
    # print(layer_1_num)
    for index_1,action_count in enumerate(subdirectories_1):

        layer_2_dir=os.path.join(waifolder, action_count)
        # print(layer_2_dir)
        entries = os.listdir(layer_2_dir)
        # 过滤出子文件夹
        subdirectories_2 = sorted_files = sorted(entries, key=lambda x: int(x.split('_')[1]))
        layer_2_num = len(subdirectories_2)
        # print(layer_2_num)
        for index_2 in range(0,len(subdirectories_2),1):
            print(
                '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print(subdirectories_2[index_2])
            frame_count=subdirectories_2[index_2]
            layer_frame = os.path.join(layer_2_dir, frame_count)
            lidu/=2
            print('ppppppppppppppppppppppppppppp')
            print(index_2)
            SDI,frame_count_SDI=main_function_seeding_metrics(layer_frame, [0, 11], 'png', 0.7, (0, 0, 200, 200), 60, 0)

            frame_count_SDI=frame_count_SDI.reshape(3,4)

            SDI_JIHE[index_1,index_2]=frame_count_SDI
        print(
            'kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
        print(index_1)
    m = np.sum(np.all(SDI_JIHE == 0, axis=(1, 2, 3)))
    n = np.sum(np.all(SDI_JIHE == 0, axis=(0, 2, 3)))

    # print('++++++++++++++++++++++++++++++++++++++++++')
    # print((m,n))
    #
    # print('++++++++++++++++++++++++++++++++++++++++++')
    for index_3 in range(layer_1_num):
        for index_4 in range(30):
            SCR_frame32 = -1*(SDI_JIHE[index_3,index_4+2] - SDI_JIHE[index_3,index_4]+1 )
            SCR_frame21 = -1 * (SDI_JIHE[index_3, index_4 +1] - SDI_JIHE[index_3, index_4])
            SCR_frame31 = -1 * (SDI_JIHE[index_3, index_4 + 2] - SDI_JIHE[index_3, index_4+1])
            SCR_frame32 = np.nan_to_num(SCR_frame32, nan=0.01)
            SCR_frame21= np.nan_to_num(SCR_frame21, nan=0.01)
            SCR_frame31= np.nan_to_num(SCR_frame31, nan=0.01)
            # SCR_frame=abs((SCR_frame31*SCR_frame32/SCR_frame21))
            SCR_frame=2.71**((10*(SDI_JIHE[index_3,index_4+1]-SDI_JIHE[index_3,index_4])))
            # SCR_frame=SCR_frame/
            # SCR_frame = 2.71**((SDI_JIHE[index_3,index_4+1]-SDI_JIHE[index_3,index_4])/SDI_JIHE[index_3,index_4+1])
            SCR_frame=np.nan_to_num(SCR_frame,nan=0.000001)
            print('---------------------jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj')
            print(index_4)

            print(SCR_frame)
            SCR_JIHE[index_3,index_4]=SCR_frame
            # print('^^^^^^^^^^^^^^^^^^^^^'+f'{index_4}')
            # print(SCR_JIHE)
            # print('^^^^^^^^^^^^^^^^^^^^^'+f'{index_4}')

        max_index = np.unravel_index(np.argmax(SCR_JIHE[index_3]), SCR_JIHE[index_3].shape)
        print(np.max(SCR_JIHE[index_3]))
        SCR_ACTION.append((index_3,) + max_index)
    print(SCR_ACTION)
    for scr_gogo in SCR_ACTION:
        list1=list(scr_gogo)
        list1[2]+=1
        list1[3] += 1
        print(list1)
    data = np.load('E:/actionfile/action_zuobiaoji.npy')
    transformed_data = np.floor((data / 200) + 1).astype(int)

    for item in transformed_data:
        item = item[::-1]
        print(item)

        # for index_3,SDI_FRAME in enumerate(SDI_JIHE):
        #     SCR_frame=(SDI_JIHE(index_3+1)-SDI_JIHE(index_3))/SDI_JIHE(index_3)
        #     SCR_JIHE.append(SCR_frame)
            # print(layer_frame)
        # while len(target) > 1:
        #     lidu/=2
        #     # frame_head = layer_2_num / 2 - lidu
        #     # frame_tail = layer_2_num / 2 + lidu
        #     # frame_head = os.path.join(layer_2_dir, subdirectories_2(frame_head))
        #     # frame_tail = os.path.join(layer_2_dir, subdirectories_2(frame_tail))
        #     # SDI_head = main_function_seeding_metrics(frame_head, [0, 11], 'png', 0.3, (0, 0, 200, 200), 50, 0)
        #     # SDI_tail = main_function_seeding_metrics(frame_tail, [0, 11], 'png', 0.3, (0, 0, 200, 200), 50, 0)
        #
        #     SCR = (SDI_tail - SDI_head) / SDI_head
        #     max_scr = np.argmax(SCR)
        #     max_scr_index = np.unravel_index(max_scr, SCR.shape)
    return SCR_ACTION

def ceshi():

        # frame_head = os.path.join(folder, subdirectories_1(frame_head))
        # frame_tail = os.path.join(folder, subdirectories_1(frame_tail))

        # SDI_head = main_function_seeding_metrics('E:/frameout/timestamp_0001/image_11', [0, 11], 'png', 0.3, (0, 0, 200, 200), 50, 0)
        SDI_tail = main_function_seeding_metrics('E:/frameout/timestamp_0001/image_12', [0, 11], 'png', 0.3, (0, 0, 200, 200), 50, 0)
        # SDI_head = np.array(SDI_head)
        # SDI_head=SDI_head.reshape(3,4)
        # SDI_tail = np.array(SDI_tail)
        # SDI_tail=SDI_tail.reshape(3,4)
        # SCR = -10*(SDI_tail - SDI_head) / SDI_head
        # SCR=SCR.reshape(3, 4)
        # max_scr = np.argmax(SCR)
        # max_scr_index = np.unravel_index(max_scr, SCR.shape)
        # max_scr_index = (max_scr_index[0] + 1, max_scr_index[1] + 1)
        # print('--------')
        # print(SCR)
        # print(SDI_head.shape)
        # print(max_scr_index)
        # print('--------')
def npy_print():
    file1_path = 'E:/actionfile/action_zuobiaoji.npy'
    file2_path = 'E:/actionfile/action_timestamps.npy'

    # 使用 numpy.load() 加载文件
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)

    # 打印输出
    print("文件1的内容:")
    print(data1)

    print("文件2的内容:")
    print(data2)


def sdiprintandplt():

    contIniFin = [0, 11]  # 图像帧范围
    ImFormat = 'png'  # 图像格式
    BinThreshold = 0.7  # 二值化阈值
    ROI = (0, 0, 200, 200)  # 感兴趣区域
    blockSizeR = 20  # 块大小
    MinValueAreaTracers = 0  # 粒子最小面积

# main_function_seeding_metrics('E:/test1/',[0, 11], 'png', 0.3, (0, 0, 200, 200), 50, 0)
# image_folder = 'E:/frame/'  # 存放.png图像的文件夹
# timestamps_file = 'E:/actionfile/action_timestamps.npy'  # 存放时间戳的.npy文件
# output_folder = 'E:/frameout/'  # 输出文件夹
# # printdata_png()
# # timestamps = np.load(timestamps_file)
# timestamps = np.load(timestamps_file, allow_pickle=False)
#
# # process_images(timestamps, image_folder, output_folder)
# SCR_go()
# # npy_print()
# # ceshi()