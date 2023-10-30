import torch
import numpy as np
from tqdm import trange
from torch import Tensor
import math

CUDA = torch.cuda.is_available()

class kmeans_core:
    def __init__(self, k, data_array, batch_size=8e5, epochs=200, all_cuda=True):
        """
        kmeans by batch
        k: number of the starting centroids
        data_array:numpy array of data
        batch_size:batch size
        epochs: max epoch iterations, if the centeroids not shifting any more, the calculation will cease before this max number
        all_cuda: do you want to move the entire array to the cuda
        
        About data loader: We didn't use dataloader. The data loader will load data entry by entry with cpu multi processor, hence losing the power of fast gpu. Matter of fact, when I use the dataloader the 92.1% of the time consumption is caused by data loader
        """
        self.k = k
        self.data_array = data_array
        self.tensor = Tensor(self.data_array,)
        self.all_cuda = all_cuda
        if all_cuda and CUDA:
            self.tensor = self.tensor.cuda()
        
        self.dim = data_array.shape[-1]
        self.data_len = data_array.shape[0]
        
        self.cent = Tensor(data_array[np.random.choice(range(self.data_len), k)])
        
        if CUDA:
            self.cent = self.cent.cuda()
            
        self.epochs = epochs
        self.batch_size = int(batch_size)
        self.iters = math.ceil(self.data_array.shape[0]/self.batch_size)
        self.index = 0
        

    def get_data(self,index):
        return self.tensor[index:index+self.batch_size,...]

    def run(self):
        for e in range(self.epochs):
            # t = trange(self.iters)
            t = self.iters

            start = self.cent.clone()
            for i in range(t):
                dt = self.get_data(self.index)
                self.index += self.batch_size
                
                if CUDA and self.all_cuda==False:
                    dt = dt.cuda()  
                self.step(dt)
                # t.set_description("🔥[epoch:%s\t iter:%s]🔥 \t🔥k:%s\t🔥distance:%.3f" % (e, i, self.k, self.distance))
            self.index=0
            
            if self.cent.size()[0] == start.size()[0]:
                if self.cent.sum().item() == start.sum().item():
                    # print("Centroids is not shifting anymore")
                    break
                    
        # t = trange(self.iters)
        t = range(self.iters)
        
        for i in t:
            dt = self.get_data(self.index)
            self.index += self.batch_size
            if CUDA and self.all_cuda==False:
                dt = dt.cuda()
            if i == 0:
                self.idx = self.calc_idx(dt)
            else:
                self.idx = torch.cat([self.idx, self.calc_idx(dt)], dim=-1)
        self.index=0
        return self.idx

    def step(self, dt):
        idx = self.calc_idx(dt)
        self.new_c(idx, dt)

    def calc_distance(self, dt):
        bs = dt.size()[0]
        distance = torch.pow(self.cent.unsqueeze(0).repeat(bs, 1, 1) - dt.unsqueeze(1).repeat(1, self.k, 1), 2).mean(
            dim=-1)
        return distance

    def calc_idx(self, dt):
        distance = self.calc_distance(dt)
        self.distance = distance.mean().item()
        val, idx = torch.min(distance, dim=-1)
        return idx

    def new_c(self, idx, dt):
        if CUDA:
            z = torch.cuda.FloatTensor(self.k, self.dim).fill_(0)
            o = torch.cuda.FloatTensor(self.k).fill_(0)
            ones = torch.cuda.FloatTensor(dt.size()[0]).fill_(1)
        else:
            z = torch.zeros(self.k, self.dim)
            o = torch.zeros(self.k)
            ones = torch.ones(dt.size()[0])
            
        ct = o.index_add(0, idx, ones)

        # slice to remove empety sum (no more such centroid)
        slice_ = (ct > 0)

        cent_sum = z.index_add(0, idx, dt)[slice_.view(-1, 1).repeat(1,self.dim)].view(-1, self.dim)
        ct = ct[slice_].view(-1, 1)

        self.cent = cent_sum / ct
        self.k = self.cent.size()[0]

class iou_km(kmeans_core):
    def __init__(self, k, data_array, batch_size=1000, epochs=200):
        super(iou_km, self).__init__(k, data_array, batch_size=batch_size, epochs=epochs)

    def calc_distance(self, dt):
        """
        calculation steps here
        dt is the data batch , size = (batch_size , dimension)
        self.cent is the centroid, size = (k,dim)
        the return distance, size = (batch_size , k), from each point's distance to centroids
        """
        bs = dt.size()[0]
        box = dt.unsqueeze(1).repeat(1, self.k, 1)
        anc = self.cent.unsqueeze(0).repeat(bs, 1, 1)

        outer = torch.max(box[..., 2:4], anc[..., 2:4])
        inner = torch.min(box[..., 2:4], anc[..., 2:4])

        inter = inner[..., 0] * inner[..., 1]
        union = outer[..., 0] * outer[..., 1]

        distance = 1 - inter / union

        return distance