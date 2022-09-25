import pandas as pd

from UserAction_Dataset import load_df, seq_len, feature_len, window_size, user_size, UserAction_Dataset
from torch.utils.data import Dataset, random_split
import torch


class UserAction_WinData(Dataset):
    def __init__(self, _tab: pd.DataFrame, set_type="train"):
        super(UserAction_WinData, self).__init__()
        self.set_type = set_type
        self.source = _tab.groupby("id")

    def __len__(self):
        return user_size

    def __getitem__(self, item):
        return self.concat_data(self.source.get_group(item))

    def concat_data(self, tar):
        return

    @classmethod
    def split_day_init(cls):
        df_temp = load_df()
        df_train = df_temp[(df_temp['day'] >= 1) & (df_temp['day'] <= 4)]
        df_test = df_temp[df_temp['day'] == 5]
        df_test = df_test[df_test['is_free'] == 1]
        return

    @classmethod
    def loading_all_init(cls):
        return cls(load_df())

    pass


if __name__ == '__main__':
    obj = UserAction_WinData.loading_all_init()
    res = obj[1]
    pass
