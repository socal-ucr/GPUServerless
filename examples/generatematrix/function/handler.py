from __future__ import print_function
from __future__ import absolute_import
from minio import Minio
import requests
import json
import uuid
import os
import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import threading
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import time
import numpy as np
import io
from six.moves import range
from math import ceil
code = """
    #include <curand_kernel.h>

    const int nstates = %(NGENERATORS)s;
    __device__ curandState_t* states[nstates];
    
    extern "C" {
    __global__ void initkernel(int seed)
    {
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;

        if (tidx < nstates) {
            curandState_t* s = new curandState_t;
            if (s != 0) {
                curand_init(seed, tidx, 0, s);
            }

            states[tidx] = s;
        }
    }

    __global__ void randfillkernel(float *values, unsigned long N)
    {
        unsigned long tidx = threadIdx.x + blockIdx.x * blockDim.x;

        if (tidx < nstates) {
            curandState_t s = *states[tidx];
            for(unsigned long i=tidx; i < N; i += blockDim.x * gridDim.x) {
                values[i] = curand_uniform(&s);
            }
            *states[tidx] = s;
        }
    }
    }
"""
cdata = 1
class GPUThread(threading.Thread):
    def __init__(self,devID,hptr):
        threading.Thread.__init__(self)
        self.devID = devID
        self.hptr = hptr
    def run(self):
        self.dev = cuda.Device(self.devID)
        self.ctx = self.dev.make_context()
        global code
        global cdata
        numThreads = 1024
        numBlocks = 30
        N = numThreads * numBlocks
        mod = SourceModule(code % { "NGENERATORS" : N },no_extern_c=True, arch='sm_61')
        init_func = mod.get_function("initkernel")
        fill_func = mod.get_function("randfillkernel")
        init_func(np.int32(time.time()), block=(numThreads,1,1), grid=(numBlocks,1,1))

        split = ceil(self.hptr.size / float(2000000000))        
        split_data = np.array_split(self.hptr,split)
        gdata = gpuarray.GPUArray(shape=split_data[0].size,dtype=np.float32)
        for data in split_data:
            fill_func(gdata, np.uint64(gdata.size), block=(numThreads,1,1), grid=(numBlocks,1,1))
            gdata.get(data)

        self.ctx.pop()
        del self.ctx

def handle(st):
    start = time.time()
    cuda.init()
    req = json.loads(st)

    mc = Minio(os.environ['minio_hostname'],
                  access_key=os.environ['minio_access_key'],
                  secret_key=os.environ['minio_secret_key'],
                  secure=False)
    width = int(req['width'])
    height = int(req['height'])
    global cdata
    cdata = np.zeros(shape=np.uint64(width*height),dtype=np.float32)
    print(cdata)
    split = 1
    if cdata.size >= 1024*30*1000:
        split = 2
    print("Num Devices: "+ str(split))
    split_data = np.array_split(cdata,split)
    gpu_thread_list = []
    index = 0
    for i in range(split):
        gpu_thread = GPUThread(i,split_data[i])
        gpu_thread.start()
        gpu_thread_list.append(gpu_thread)
        index += split_data[i].size

    for t in gpu_thread_list:
        t.join()
    print(cdata)
    end = time.time()
    print(end-start)
    start = time.time()
    buf = cdata.tobytes()
    file_name = get_temp_file()
    mc.put_object('cudata',file_name,io.BytesIO(buf),len(buf))
    end = time.time()
    print(end-start)
    return file_name


def get_temp_file():
    uuid_value = str(uuid.uuid4())
    return uuid_value
