# Adapted from https://github.com/yjw1029/Efficient-FedRec/blob/839f967c1ed1c0cb0b1b4d670828437ffb712f29/preprocess/adressa_raw.py

from typing import List

import numpy as np


class UserInfo():
    def __init__(
            self,
            train_day: int,
            test_day: int
            ) -> None:

        self.hist_news = []
        self.hist_time = []

        self.train_news = []
        self.train_time = []

        self.test_news = []
        self.test_time = []

        self.train_day = train_day
        self.test_day = test_day

    def update(self, nindex, click_time, day):
        if day == self.train_day:
            self.train_news.append(nindex)
            self.train_time.append(click_time)
        elif day == self.test_day:
            self.test_news.append(nindex)
            self.test_time.append(click_time)
        else:
            self.hist_news.append(nindex)
            self.hist_time.append(click_time)

    def sort_click(self):
        self.train_news = np.array(self.train_news, dtype='int32')
        self.train_time = np.array(self.train_time, dtype='int32')

        self.test_news = np.array(self.test_news, dtype='int32')
        self.test_time = np.array(self.test_time, dtype='int32')

        self.hist_news = np.array(self.hist_news, dtype='int32')
        self.hist_time = np.array(self.hist_time, dtype='int32')

        order = np.argsort(self.train_time)
        self.train_news = self.train_news[order]
        self.train_time = self.train_time[order]

        order = np.argsort(self.test_time)
        self.test_news = self.test_news[order]
        self.test_time = self.test_time[order]

        order = np.argsort(self.hist_time)
        self.hist_news = self.hist_news[order]
        self.hist_time = self.hist_time[order]

