import numpy as np
import monai.transforms as transforms

class CustomNormalize:
    def __init__(self, keys_group1, a_min_group1, a_max_group1, b_min_group1, b_max_group1,
                 keys_group2, a_min_group2, a_max_group2, b_min_group2, b_max_group2):
        self.keys = keys_group1 + keys_group2
        self.normalizers = {}
        
        for key, amin, amax, bmin, bmax in zip(keys_group1, a_min_group1, a_max_group1, b_min_group1, b_max_group1):
            self.normalizers[key] = transforms.ScaleIntensityRanged(
                a_min=amin, a_max=amax, b_min=bmin, b_max=bmax, clip=True
            )
            
        for key, amin, amax, bmin, bmax in zip(keys_group2, a_min_group2, a_max_group2, b_min_group2, b_max_group2):
            self.normalizers[key] = transforms.ScaleIntensityRanged(
                a_min=amin, a_max=amax, b_min=bmin, b_max=bmax, clip=True
            )

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.normalizers[key](data[key])
        return data
    

class AddRicianNoise(transforms.MapTransform):
    def __init__(self, keys, noise_std):
        super().__init__(keys)
        self.noise_std = noise_std

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            noise_real = np.random.normal(loc=0, scale=self.noise_std, size=image.shape)
            noise_imag = np.random.normal(loc=0, scale=self.noise_std, size=image.shape)
            noisy_image = np.sqrt((image + noise_real) ** 2 + noise_imag ** 2)
            d[key] = noisy_image
        return d
    
