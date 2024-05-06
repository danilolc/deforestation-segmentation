
import os
import pandas as pd
import numpy as np
import torch

class RandomSimetry():
    def __call__(self, sample):
        if torch.rand(1) > 0.5:
            sample = torch.rot90(sample, 1, [1,2])

        if torch.rand(1) > 0.5:
            sample = torch.flip(sample, [1])

        if torch.rand(1) > 0.5:
            sample = torch.flip(sample, [2])

        return sample

class ChannelJitter():
    def __init__(self, sum_fac, mul_fac):
        self.sum_fac = sum_fac
        self.mul_fac = mul_fac

    def __call__(self, sample):
        channels = sample.shape[0]
        
        sum_rnd = torch.randn(channels, device=sample.device) * self.sum_fac
        sum_rnd[-1] = 0  # last channel is the label

        mul_rnd = torch.randn(channels, device=sample.device) * self.mul_fac + 1
        mul_rnd[-1] = 1

        # sample shape is (channels, height, width)
        sample = sample * mul_rnd.view(channels, 1, 1)
        sample = sample + sum_rnd.view(channels, 1, 1)
        return sample


class RandomNoise():
    def __init__(self, std,):
        self.std = std

    def __call__(self, sample):
        std = torch.rand(1, device=sample.device) * self.std

        noise = torch.randn(sample.shape, device=sample.device) * std
        noise[-1] = 0

        return sample + noise

ImMean = torch.tensor([-0.0545, -0.0508, -0.0239, -0.0395,  0.2755,  0.1047,  0.0010, 0])
ImSTD  = torch.tensor([ 0.0082,  0.0098,  0.0163,  0.0204,  0.0448,  0.0482,  0.0317, 1])

def get_values(filename):
    fn = filename.split("_")

    lat = float(fn[1])
    lon = float(fn[2])
    year = int(fn[3])
    month = int(fn[4])
    day = int(fn[5])
    
    mean = fn[6].split(".")
    mean = int(mean[0]) + int(mean[1]) / 100

    # return a dictionary
    return {"lat": lat, "lon": lon, "year": year, "month": month, "day": day, "mean": mean, "filename": filename}

def load_npy(filename):
    img = np.load(filename)

    image = torch.from_numpy(img[:-1, :, :])
    label = torch.from_numpy(img[-1, :, :])

    return image, label

def load_df(path):

    images = os.listdir(path)
    images = [get_values(image) for image in images]

    return pd.DataFrame(images)