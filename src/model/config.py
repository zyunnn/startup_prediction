class Config:
    def __init__(self,
                 X_path = "src/data/X.npy",
                 Y_path = "src/data/Y.npy", 
                 sample_length=6,
                 X_input_filtering_length = 19,
                 sampling_ratio = 10):
        self.X_path = X_path
        self.Y_path = Y_path
        self.sample_length = sample_length
        self.X_input_filtering_length = X_input_filtering_length
        self.sampling_ratio = sampling_ratio


