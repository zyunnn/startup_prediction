import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from config import Config


class DataLoader():
    def __init__(self, config):
        self._X = np.load(config.X_path,allow_pickle=True)
        self._Y = np.load(config.Y_path,allow_pickle=True)
        self.sample_length = config.sample_length
        self.positive_startpoints = []
        self.X_input_filtering_length = config.X_input_filtering_length
        
        for i in range(self._Y.shape[0]):
            for j in range(self._Y.shape[1] - self.sample_length):
                if self._Y[i, j+self.sample_length-1] > 0 and (not np.all(self._X[i,j:j+self.sample_length,:self.X_input_filtering_length]==0)):
                    self.positive_startpoints.append((i, j))    

        # Number of samples to be extracted
        num_pos = len(self.positive_startpoints)
        num_rand = config.sampling_ratio*num_pos

        # Get index of positive samples
        list_X = [self._X[pair[0], pair[1]:pair[1]+self.sample_length] for pair in self.positive_startpoints]
        list_Y = [self._Y[pair[0], pair[1]+self.sample_length-1] for pair in self.positive_startpoints]

        rand_start_row = np.random.randint(self._Y.shape[0], size=num_rand)
        rand_start_col = np.random.randint(self._Y.shape[1] - self.sample_length, size=num_rand)
        
        # Combine positive and negative samples
        rand_start_points = np.stack([rand_start_row, rand_start_col]).transpose()
        rand_list_X = [self._X[list(pair)[0], list(pair)[1]:list(pair)[1]+self.sample_length] for pair in rand_start_points]
        rand_list_Y = [self._Y[list(pair)[0], list(pair)[1] + self.sample_length-1] for pair in rand_start_points]

        list_X.extend(rand_list_X)
        list_Y.extend(rand_list_Y)

        self._sequence_X = np.stack(list_X)
        self._sequence_Y = np.stack(list_Y)
        
        # Split data into train (+ dev) and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._sequence_X, self._sequence_Y, test_size=0.2)
    
    def get_positive_points(self):
        return self.positive_startpoints

    def get_sequence_Y(self):
        return self._sequence_Y

    def get_batch_train(self):
        return self.X_train, self.y_train

    def get_batch_test(self):
        return self.X_test, self.y_test


if __name__ == '__main__':
    conf = Config()
    d = DataLoader(conf)


