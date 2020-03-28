import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.preprocessing.sequence.TimeseriesGenerator):
    def __init__(self, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=8,
                 n_outputs=24):

        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index - n_outputs  # Needed to avoid overflow of target
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.n_outputs = n_outputs

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __len__(self):
        return (self.end_index - self.start_index +
                self.batch_size * self.stride) // (self.batch_size * self.stride)

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        samples = np.array([self.data[row - self.length:row:self.sampling_rate]
                            for row in rows])
        try:
            targets = np.array([self.targets[row:row + self.n_outputs] for row in rows]).reshape(len(rows),
                                                                                                 self.n_outputs)  # Not use batch_size
        except:
            print(np.array([self.targets[row:row + self.n_outputs] for row in rows]).shape)
        # targets_dec = np.array([[0, *self.targets[row:row+self.n_outputs-1]] for row in rows]).reshape(self.batch_size, self.n_outputs, 1)
        if self.reverse:
            return samples[:, ::-1, ...], targets

        return samples, targets
