from email import message
import json
import time
from datetime import datetime
import numpy as np

fmt = '%Y-%m-%d %H:%M:%S'
refer_time = '2020-01-02 00:00:00'
userinfo_dtype = np.dtype(
    [("id", "i4"), ("time", "f4"), ("day", "i1"), ("hours", "i2"), ("minutes", "i2"), ("screen_lock_status", "i1"),
     ("network_status", "i1"),
     ("battery_charge_status", "i1"), ("screen_status", "i1"), ("battery_level", "i1"),
     ("is_free", "i1"), ("idle", "i1")])


class userinfo_entry:
    # 用户的id，用来标识某个用户
    user_id = -1
    # 手机型号
    # phone_model = -1
    # 从两周的第0天开始计时，是一个时间戳
    time = -1
    # 一周中的第几天
    day = -1
    # 24小时制
    hours = -1
    # 多少分钟
    minutes = -1
    # 屏幕是否解锁，0表示未解锁，1表示解锁
    screen_lock_status = -1
    # 网络状态，0不可用, 1表示wifi
    network_status = -1
    # 电池充电状态，0表示未充电，1表示充电
    battery_charge_status = 0
    # 屏幕状态，0表示屏幕关闭，1表示屏幕打开
    screen_status = -1
    # 电池电量，是一个百分数
    battery_level = -1
    # 设备是否空闲
    is_free = 0
    # 10分钟内是否空闲
    idle = 0

    def __str__(self):
        return (
                "user_id:%d time:%d day:%d hours:%d minutes:%d screen_lock_status:%d network_status:%d "
                "battery_charge_status:%d " "screen_status:%d " "battery_level:%f is_free:%d"
                % (
                    self.user_id, self.time, self.day, self.hours, self.minutes, self.screen_lock_status,
                    self.network_status,
                    self.battery_charge_status,
                    self.screen_status, self.battery_level, self.is_free))

    def is_valid(self):
        if (self.user_id == -1 or self.time == -1 or self.screen_lock_status == -1
                or self.network_status == -1 or self.battery_charge_status == -1 or self.screen_status == -1 or self.battery_level == -1):
            return False
        else:
            return True

    def numpy(self):
        return np.array(
            (self.user_id, self.time, self.day, self.hours, self.minutes, self.screen_lock_status, self.network_status,
             self.battery_charge_status, self.screen_status, self.battery_level, self.is_free, self.idle),
            dtype=userinfo_dtype)


def get_days(end_time, begin_time=0):
    times = end_time - begin_time
    day = times / (24 * 3600)
    day = int((day + 4) % 7)
    leave1 = times % (24 * 3600)
    hours = int(leave1 / 3600)
    leave2 = leave1 % 3600
    minutes = int(leave2 / 60)
    return day, hours, minutes


def build_dataset():
    # user_array = np.array([], dtype=userinfo_dtype)
    userinfo_list = []
    with open('./state_traces.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        count1 = 0
        count2 = 0
        for idx, user_value in enumerate(d.values()):
            user_list = []
            entry = userinfo_entry()
            entry.user_id = int(user_value["guid"])
            message = user_value["messages"].split("\n")
            for mes in message:
                if mes.strip() == "":
                    continue
                t, s = mes.strip().split("\t")
                t = t.strip()
                s = s.strip()
                s = s.lower()
                time_stamp = time.mktime(datetime.strptime(t, fmt).timetuple()) - time.mktime(
                    datetime.strptime(refer_time, fmt).timetuple())
                entry.day, entry.hours, entry.minutes = get_days(time_stamp)
                entry.time = time_stamp
                if s == 'battery_charged_on':
                    entry.battery_charge_status = 1
                    charged = True
                elif s == 'battery_charged_off':
                    entry.battery_charge_status = 0
                elif s == 'wifi':
                    entry.network_status = 1
                elif s == 'unknown' or s == '4g' or s == '3g' or s == '2g' or s == '5g':
                    entry.network_status = 0
                elif s == 'screen_on':
                    entry.screen_status = 1
                elif s == 'screen_off':
                    entry.screen_status = 0
                elif s == 'screen_lock':
                    entry.screen_lock_status = 0
                elif s == 'screen_unlock':
                    entry.screen_lock_status = 1
                elif s[-1] == '%':
                    entry.battery_level = int(float(s[:-1]))
                else:
                    print(s)
                if entry.screen_lock_status == 0 and entry.network_status == 1 and entry.battery_charge_status == 1:
                    entry.is_free = 1
                else:
                    entry.is_free = 0
                if entry.is_valid():
                    userinfo_list.append(entry.numpy())
                    if entry.user_id == idx:
                        user_list.append(entry.numpy())

            user1info_array = np.array(user_list, dtype=userinfo_dtype)
            np.save(f"./temp/users/trace_user{idx}.npy", user1info_array, allow_pickle=True, fix_imports=True)
            print(f"idx:{idx}")

        userinfo_array = np.array(userinfo_list, dtype=userinfo_dtype)
        np.save("./temp/trace.npy", userinfo_array, allow_pickle=True, fix_imports=True)
        print(userinfo_array.shape)
        print(userinfo_array, np.sum(userinfo_array['is_free']))


def get_label(trace: object) -> object:
    label_list = []
    for id in range(1000):
        cur_trace = a[trace[:]["id"] == id]
        trace_len = len(cur_trace)
        label = np.zeros(trace_len)
        for i in range(trace_len):
            for j in range(i + 1, trace_len):
                if cur_trace[j]["screen_lock_status"] == 0 and cur_trace[j]["network_status"] == 1 and cur_trace[j]["battery_charge_status"] == 1:
                    if cur_trace[j]["time"] - cur_trace[i]["time"] >= 60:
                        label[i] = 1
                        break
                    else:
                        continue
                else:
                    label[i] = 0
                    break
        # print(label)
        label_list.extend(np.asarray(label))
    return np.array(label_list)


# build_dataset()

if __name__ == "__main__":
    build_dataset()
    a = np.load("./temp/trace.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)
    print(a)
    label_arr = get_label(a)
    # # a[:]['idle'] = label_arr
    #
    np.save("temp/label_arr.npy", label_arr, allow_pickle=True, fix_imports=True)
    #
    print("finished")

    # import matplotlib.pyplot as plt
    # plt.hist(label_arr)
    # plt.show()
    #
    # print(label_arr.shape)

    # max_id = np.max(a[:]["id"])
    # min_id = np.min(a[:]["id"])
    # a[:]["id"] = (a[:]["id"] - min_id) / (max_id - min_id)
    # max_time = np.max(a[:]["time"])
    # min_time = np.min(a[:]["time"])
    # a[:]["time"] = (a[:]["time"] - min_time) / (max_time - min_time)
    # a[:]["battery_level"] = a[:]["battery_level"] / 100
    # batch_var_x = []
    # batch_var_y = []

    # for id in range(100):

    # print(a.shape)
    pass
