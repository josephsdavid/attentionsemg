import tensorflow.keras as keras
import numpy as np
import abc
from .helpers import read_file_validation, pad_along_axis
from .augmentors import window_roll, roll_labels
from . import preprocessors as pp

class Loader(abc.ABC):
    @abc.abstractmethod
    def read_data(self):
        pass

    @abc.abstractmethod
    def process_data(self):
        pass

    @abc.abstractmethod
    def augment_data(self):
        pass



class PreValidationLoader(Loader):
    def __init__(self, path: str, process_fns: list, augment_fns: list, scale=False, step = 5, window_size = 52):
        self.path = path
        self.processors = process_fns
        self.augmentors = augment_fns
        self.read_data()
        self.process_data()
        self.augment_data(step, window_size)
        self.emg = np.moveaxis(np.concatenate(self.emg,axis=0),2,1)
        if scale:
            self.emg = pp.scale(self.emg)

    def _read_group_to_lists(self):
        # grab in all the male candidates
        n_classes=7
        res = []
        labels = []
        trials = range(n_classes*4)
        for candidate in range(12):
            man = [read_file_validation(self.path + '/Male' + str(candidate) + '/training0/classe_%d.dat' %i) for i in trials]
            # list addition is my new favorite python thing
            labs = [t % n_classes for t in trials]
            res += man
            labels += labs

        # and all the female candidates
        for candidate in range(7):
            woman = [read_file_validation(self.path + '/Female' + str(candidate) + '/training0/classe_%d.dat' %i) for i in trials]
            labs = [t % n_classes for t in trials]
            res += woman
            labels += labs

        return res, labels

    def read_data(self):
        self.emg, self.labels = self._read_group_to_lists()
        self.emg = [pad_along_axis(x, 1000) for x in self.emg]

    def process_data(self):
        for f in self.processors:
            self.emg = [f(x) for x in self.emg]

    def augment_data(self, step, window_size):
        #for f in self.augmentors:
        #    self.emg, self.labels = f(self.emg, self.labels)

        self.emg = [window_roll(x, step, window_size) for x in self.emg]
        self.labels = roll_labels(self.emg, self.labels)


class PreTrainLoader(Loader):
    def __init__(self, path: str, process_fns: list, augment_fns: list, scale=False, step =5, window_size=52):
        self.path = path
        self.processors = process_fns
        self.augmentors = augment_fns
        self.read_data()
        self.process_data()
        self.augment_data(step, window_size)
        self.emg = np.moveaxis(np.concatenate(self.emg,axis=0),2,1)
        if scale:
            self.emg = pp.scale(self.emg)

    def _read_group_to_lists(self):
        res = []
        labels = []
        trials = range(7*4)
        for instance in ['training0', 'Test0', 'Test1']:
            for candidate in range(15):
                man = [read_file_validation(self.path + '/Male' + str(candidate) + '/' + instance + '/classe_%d.dat' %i) for i in trials]
                # list addition is my new favorite python thing
                labs = [t % 7 for t in trials]
                res += man
                labels += labs
                # and all the female candidates
            for candidate in range(2):
                woman = [read_file_validation(self.path + '/Female' + str(candidate) + '/' + instance + '/classe_%d.dat' %i) for i in trials]
                labs = [t % 7 for t in trials]
                res += woman
                labels += labs
        return res, labels

    def read_data(self):
        self.emg, self.labels = self._read_group_to_lists()
        self.emg = [pad_along_axis(x, 1000) for x in self.emg]

    def process_data(self):
        for f in self.processors:
            self.emg = [f(x) for x in self.emg]

    def augment_data(self, step, window_size):
        pass
        #for f in self.augmentors:
        #    self.emg, self.labels = f(self.emg, self.labels)

        self.emg = [window_roll(x, step, window_size) for x in self.emg]
        self.labels = roll_labels(self.emg, self.labels)

# x = ValidationLoader("../../PreTrainingDataset", [pp.butter_highpass_filter], [aa.add_noise])



