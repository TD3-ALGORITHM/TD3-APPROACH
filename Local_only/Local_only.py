import math
import random

import numpy as np


class UAVEnv(object):
    height = ground_length = ground_width = 100  # 场地长宽均为100m，UAV飞行高度也是 --#The length and width of the venue are both 100m, and the flight height of the UAV is also 100m.
    sum_task_size = 60 * 1048576  # 总计算任务60 Mbits --#Total computing task 60 Mbits
    loc_uav = [50, 50]
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6  # 带宽1MHz --#Bandwidth 1MHz
    p_noisy_los = 10 ** (-13)  # 噪声功率-100dBm  --#Noise power -100dBm
    p_noisy_nlos = 10 ** (-11)  # 噪声功率-80dBm  --#Noise power -80dBm
    flight_speed = 50.  # 飞行速度50m/s --#Flight speed 50m/s
    f_ue = 6e8  # UE的计算频率0.6GHz --The calculation frequency of UE is 0.6GHz
    f_uav = 12e8  # UAV的计算频率1.2GHz --The calculation frequency of UAV is 1.2GHz
    r = 10 ** (-27)  # 芯片结构对cpu处理的影响因子 --Influence factor of chip structure on cpu processing
    s = 1000  # 单位bit处理所需cpu圈数1000  --# The number of cpu circles required for unit bit processing is 1000
    p_uplink = 0.1  # 上行链路传输功率0.1W  --Uplink transmission power 0.1W
    # alpha0 = -30  # 距离为1m时的参考信道增益-30dB   --Reference channel gain -30dB at a distance of 1m
    alpha0 = 1e-5  # 距离为1m时的参考信道增益-50dB = 1e-5  --Reference channel gain -50dB = 1e-5 at a distance of 1m
    T = 200  # 周期200s  --Cycle 200s
    delta_t = 5  # 1s飞行, 后4s用于悬停计算  --1s flight, the last 4s is used for hover calculation
    slot_num = int(T / delta_t)  # 40个间隔 --40 intervals
    m_uav = 9.65  # uav质量/kg --uav mass/kg
    e_battery_uav = 500000  # uav电池电量 --uav battery power: 500kJ. ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning

    #################### ues ####################
    M = 4  # UE数量 --Number of UEs
    block_flag_list = np.random.randint(0, 2, M)  # 4个ue，ue的遮挡情况 --4 ue, the occlusion of ue
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])  # 位置信息:x在0-100随机 --Position information: x is random from 0-100
    # task_list = np.random.randint(1048576, 2097153, M)    # 随机计算任务1~2Mbits --Random computing task 1~2Mbits
    task_list = np.random.randint(1572864, 2097153, M)  # 随机计算任务1.5~2Mbits --Random computing task 1.5~2Mbits
    # ue位置转移概率 --ue position transition probability
    # 0:位置不变; 1:x+1,y; 2:x,y+1; 3:x-1,y; 4:x,y-1 --0: the position remains unchanged
    loc_ue_trans_pro = np.array([[.6, .1, .1, .1, .1],
                                 [.6, .1, .1, .1, .1],
                                 [.6, .1, .1, .1, .1],
                                 [.6, .1, .1, .1, .1]])

    action_bound = [-1, 1]  # 对应tahn激活函数 --Corresponding tahn activation function
    action_dim = 4  # 第一位表示服务的ue id;中间两位表示飞行角度和距离；后1位表示目前服务于UE的卸载率 --The first digit represents the ue id of the service; the middle two digits represent the flight angle and distance; the last digit represents the unloading rate currently serving the UE
    state_dim = 4 + M * 4  # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag

    def __init__(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.start_state = np.append(self.e_battery_uav, self.loc_uav)
        self.start_state = np.append(self.start_state, self.sum_task_size)
        self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
        self.start_state = np.append(self.start_state, self.task_list)
        self.start_state = np.append(self.start_state, self.block_flag_list)
        self.state = self.start_state

    def reset(self):
        self.reset_env()
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self._get_obs()

    def reset_env(self):
        self.sum_task_size = 100 * 1048576  # 总计算任务60 Mbits --Total computing tasks 60
        self.e_battery_uav = 500000  # uav电池电量: 500kJ --uav battery level:
        self.loc_uav = [50, 50]
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])  # 位置信息:x在0-100随机 --Position information: x is random from 0-100
        self.reset_step()

    def reset_step(self):
        # self.task_list = np.random.randint(1572864, 2097153, self.M)  # 随机计算任务 1.5~2Mbits --random computing task -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        # self.task_list = np.random.randint(2097152, 2621441, self.M)  # 随机计算任务1.5~2Mbits--random computing task -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        self.task_list = np.random.randint(2621440, 3145729, self.M)  # 随机计算任务1.5~2Mbits --random computing task -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        # self.task_list = np.random.randint(3145728, 3670017, self.M)  # 随机计算任务1.5~2Mbits --random computing task  -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        # self.task_list = np.random.randint(3670016, 4194305, self.M)  # 随机计算任务1.5~2Mbits--random computing task  -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        self.block_flag_list = np.random.randint(0, 2, self.M)  # 4个ue --4 ue，ue的遮挡情况 --The occlusion of ue

    def _get_obs(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self.state

    def step(self):  # 0: 选择服务的ue编号  --Select the ue number of the service; 1: 方向theta --direction theta;; 2: --distance d; 3: offloading ratio
        step_redo = False
        is_terminal = False
        ue_id = np.random.randint(0, self.M)

        theta = 0  # 角度 --angle
        offloading_ratio = 0  # ue卸载率  --ue uninstall rate
        task_size = self.task_list[ue_id]
        block_flag = self.block_flag_list[ue_id]

        # 飞行距离 --flight distance
        dis_fly = 0  # 1s飞行距离 --flight distance
        # 飞行能耗 --flight energy consumption
        e_fly = (dis_fly / (self.delta_t * 0.5)) ** 2 * self.m_uav * (
                self.delta_t * 0.5) * 0.5  # ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning

        # UAV飞行后的位置 --position after flight
        dx_uav = dis_fly * math.cos(theta)
        dy_uav = dis_fly * math.sin(theta)
        loc_uav_after_fly_x = self.loc_uav[0] + dx_uav
        loc_uav_after_fly_y = self.loc_uav[1] + dy_uav

        # 服务器计算耗能 --Server Computing Power Consumption
        t_server = offloading_ratio * task_size / (self.f_uav / self.s)  # 在UAV边缘服务器上计算时延 --Computing Latency on UAV Edge Server
        e_server = self.r * self.f_uav ** 3 * t_server  # 在UAV边缘服务器上计算耗能 --Calculate energy consumption on UAV edge server

        if self.sum_task_size == 0:  # 计算任务全部完成 --Computational tasks are all completed
            is_terminal = True
            # file_name = 'output.txt'
            # with open(file_name, 'a') as file_obj:
            #     file_obj.write("\n======== This episode is done ========")  # 本episode结束 --This episode ends
            reward = 0
        elif self.sum_task_size - self.task_list[ue_id] < 0:  # 最后一步计算任务和ue的计算任务不匹配 --The calculation task of the last step does not match the calculation task of ue
            self.task_list = np.ones(self.M) * self.sum_task_size
            reward = 0
            step_redo = True
        elif loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > self.ground_width or loc_uav_after_fly_y < 0 or loc_uav_after_fly_y > self.ground_length:  # uav位置不对 --The uav position is wrong
            reward = -100
            step_redo = True
        elif self.e_battery_uav < e_fly:  # uav电量不能支持飞行 --uav power can not support flight
            reward = -100
        elif self.e_battery_uav - e_fly < e_server:  # uav电量不能支持计算 --uav power can not support calculation
            reward = -100
        else:  # 电量支持飞行,且计算任务合理,且计算任务能在剩余电量内计算 --The power supports flight, and the calculation tasks are reasonable, and the calculation tasks can be calculated within the remaining power
            delay = self.com_delay(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                   offloading_ratio, task_size, block_flag)  # 计算delay --calculate
            reward = delay
            # 更新下一时刻状态 --Update the status of the next moment
            self.e_battery_uav = self.e_battery_uav - e_fly - e_server  # uav 剩余电量 --remaining battery
            self.sum_task_size -= self.task_list[ue_id]  # 剩余任务量 --Remaining tasks
            for i in range(self.M):  # ue随机移动 -ue moves randomly
                tmp = np.random.rand()
                if 0.6 < tmp <= 0.7:
                    self.loc_ue_list[i] += [0, 1]
                elif 0.7 < tmp <= 0.8:
                    self.loc_ue_list[i] += [1, 0]
                elif 0.8 < tmp <= 0.9:
                    self.loc_ue_list[i] += [0, -1]
                elif 0.9 < tmp <= 1:
                    self.loc_ue_list[i] += [-1, 0]
                else:
                    self.loc_ue_list[i] += [0, 0]
                np.clip(self.loc_ue_list[i], 0, 100)
            # self.task_list = np.random.randint(1048576, 2097153, self.M)  # ue随机计算任务1~2Mbits --ue random computing task 1
            self.reset_step()

            # # 记录UE花费
            # file_name = 'output.txt'
            # # file_name = 'output_' + str(len(self.UE_loc_list)) + 'UE_DDPG.txt'
            # with open(file_name, 'a') as file_obj:
            #     file_obj.write("\nUE-" + '{:d}'.format(ue_id) + ", task size: " + '{:d}'.format(
            #         int(task_size)) + ", offloading ratio:" + '{:.2f}'.format(offloading_ratio))
            #     file_obj.write("\ndelay:" + '{:.2f}'.format(delay))
            #     file_obj.write("\nUAV hover loc:" + "[" + '{:.2f}'.format(loc_uav_after_fly_x) +
            #                    ', ' + '{:.2f}'.format(loc_uav_after_fly_y) + ']')  # 输出保留两位结果 --Output retains two digits of the result

        return reward, is_terminal, step_redo

    # 计算花费 --Calculate cost
    def com_delay(self, loc_ue, loc_uav, offloading_ratio, task_size, block_flag):
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)  # 信道增益 --channel gain
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)  # 上行链路传输速率bps --Uplink transmission rate bps
        t_tr = offloading_ratio * task_size / trans_rate  # 上传时延,1B=8bit --Upload delay, 1B=8bit
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)  # 在UAV边缘服务器上计算时延 --Computing Latency on UAV Edge Server
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s)  # 本地计算时延 --Local Computing Latency
        return max([t_tr + t_edge_com, t_local_com])


def diff_bandwidth():
    for k in range(10):
        delays_list = []
        for j in range(1, 11, 1):
            env = UAVEnv()
            env.reset()
            env.B = j * 10 ** 6  # 带宽nMHz --Bandwidth nMHz
            costs = 0
            i = 0
            while i < env.slot_num:
                delay, is_terminal, step_redo = env.step()
                costs += delay
                if step_redo:
                    continue
                if is_terminal or i == env.slot_num - 1:
                    delays_list.append(eval("{:.4f}".format(costs)))
                    break
                i = i + 1
        print(np.array(delays_list))


def diff_task_size():
    delays_list = []
    for k in range(10):
        # 不同带宽：1 - 10 MHz，记录10次，绘图时再取均值 --Different bandwidth: 1 - 10 MHz, record 10 times, take the average value when drawing
        env = UAVEnv()
        env.reset()
        costs = 0
        i = 0
        while i < env.slot_num:
            delay, is_terminal, step_redo = env.step()
            costs += delay
            if step_redo:
                continue
            if is_terminal or i == env.slot_num - 1:
                delays_list.append(eval("{:.4f}".format(costs)))
                break
            i = i + 1
    print(np.mean(delays_list))

def diff_f_ue():
    delays_list = []
    for k in range(20):
        # 不同UE的计算能力. 记录10次，绘图时再取均值 --Computing power of different UEs. Record 10 times, and take the average value when drawing
        env = UAVEnv()
        env.reset()
        costs = 0
        i = 0
        while i < env.slot_num:
            delay, is_terminal, step_redo = env.step()
            costs += delay
            if step_redo:
                continue
            if is_terminal or i == env.slot_num - 1:
                delays_list.append(costs)
                break
            i = i + 1
    print(np.around(np.mean(delays_list), 4))

if __name__ == '__main__':
    diff_f_ue()
    # diff_bandwidth()
    # diff_task_size()

'''
不同带宽条件：--Different bandwidth conditions:
[104.8576 104.8576 104.8576 104.8576 104.8576 104.8576 104.8576 104.8576 104.8576 104.8576]

'''