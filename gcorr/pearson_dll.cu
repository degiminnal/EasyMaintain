#include <cuda.h>
#include <iostream>

using namespace std;

#define M 9440

#define get_bid() (blockIdx.x)
#define get_tid() (threadIdx.x)

extern "C" __declspec(dllexport) int get_m(int&);
extern "C" __declspec(dllexport) int pearson(int, int, float*, float*, float*, float*, float*, float*, float*);


int m, n;                  // 基因表达数据的维度，m是基因数(限制：56640)，n是基因的特征维度，即细胞数
float *a;                  // CPU存储的单细胞表达数据数组，维度为 m*n
float *p;                  // GPU存储的单细胞表达数据数组
float *dstds;              // GPU内存储基因的标准差
float *dcorr0;             // GPU内储存计算结果0
float *dcorr1;             // GPU内储存计算结果1
float *dcorr2;             // GPU内储存计算结果2
float *dcorr3;             // GPU内储存计算结果3
float *dcorr4;             // GPU内储存计算结果4
float *dcorr5;             // GPU内储存计算结果5
float *corr_buf;           // CPU内储存计算结果的缓冲区
size_t pitch_p, pitch_c;   // GPU行间地址距离


__device__ void calc(volatile float* sdata, int tid, float* dx, float* dy, int n)
{
    sdata[tid] = 0;
    for(int idx=tid; idx<n; idx+=32){
        sdata[tid] += dx[idx] * dy[idx];
    }
    for(int t=16;t>=1;t/=2){
        sdata[tid] += sdata[tid+t];
    }
}

__device__ float* getptr(int r, float *dcorr0, float *dcorr1, float *dcorr2, float *dcorr3, float *dcorr4, float *dcorr5, int pitch)
{
    int a=r/M, b=r%M;
    if(a==0) return (float*)((char*)dcorr0 + b * pitch );
    if(a==1) return (float*)((char*)dcorr1 + b * pitch );
    if(a==2) return (float*)((char*)dcorr2 + b * pitch );
    if(a==3) return (float*)((char*)dcorr3 + b * pitch );
    if(a==4) return (float*)((char*)dcorr4 + b * pitch );
    if(a==5) return (float*)((char*)dcorr5 + b * pitch );
    return nullptr;
}


__global__ void add(int m, int n, float *p, int pitch_p, float *dcorr0, float *dcorr1, float *dcorr2, float *dcorr3, float *dcorr4, float *dcorr5, int pitch_c, float *dstds)
{
    __shared__ float sdata[64];
    int tid = get_tid(), bid = get_bid();

    float *dx, *dy;
    for(int i=bid;i<(m+1)/2;i+=gridDim.x){
        dx = (float*)((char*)p + i * pitch_p);
        for(int j=0;j<=i;j++){
            dy = (float*)((char*)p + j * pitch_p);
            calc(sdata, tid, dx, dy, n);
            if(tid==0) getptr(i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[j]=sdata[0];
        }        
        dx = (float*)((char*)p + (m - 1 - i) * pitch_p);
        for(int j=0;j<m-i;j++){
            dy = (float*)((char*)p + j * pitch_p);
            calc(sdata, tid, dx, dy, n);
            if(tid==0) getptr(m-1-i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[j]=sdata[0];
        }
        if(tid==0) {
            dstds[i] = sqrt(getptr(i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[i]);
            dstds[m-1-i] = sqrt(getptr(m-1-i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[m-1-i]);
        }
    }
}

__global__ void div(int m, float *dcorr0, float *dcorr1, float *dcorr2, float *dcorr3, float *dcorr4, float *dcorr5, int pitch_c, float *dstds)
{
    int bid = get_bid();
    for(int i=bid;i<m;i+=gridDim.x){
        for(int j=0;j<=i;j++){
            getptr(j, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[i] =
            getptr(i, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c)[j] /= (dstds[i] * dstds[j]);
        }
    }
}

extern "C" {
    int get_m(int& _m){_m=M;return 0;}
    int pearson(int _m, int _n, float* _a, float* _corr0, float* _corr1, float* _corr2, float* _corr3, float* _corr4, float* _corr5) {
        m = _m, n = _n, a = _a;

        for(int i=0;i<m;i++){                                                         // 元素都减去其列均值，得到 delta 数组
            float sum = 0;
            for(int j=0;j<n;j++){
                sum += a[i*n+j];
            }
            float mean = sum / n;
            for(int j=0;j<n;j++){
                a[i*n+j] -= mean;
            }
        }

        cudaMalloc((void**)&dstds, sizeof(float)*m);                                   // GPU存储标准差的数组   
        cudaMallocPitch((void**)&p, &pitch_p, sizeof(float)*n, m);                     // GPU存放基因表达数据的数组
        cudaMallocHost((void**)&corr_buf, sizeof(float)*m*M);                          // CPU存储计算结果的内存缓存区
        if(m>0)   cudaMallocPitch((void**)&dcorr0, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存0 
        if(m>1*M) cudaMallocPitch((void**)&dcorr1, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存1   
        if(m>2*M) cudaMallocPitch((void**)&dcorr2, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存2 
        if(m>3*M) cudaMallocPitch((void**)&dcorr3, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存3 
        if(m>4*M) cudaMallocPitch((void**)&dcorr4, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存4
        if(m>5*M) cudaMallocPitch((void**)&dcorr5, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存5

        cudaMemcpy2D(p, pitch_p, a, sizeof(float)*n, sizeof(float)*n, m, cudaMemcpyHostToDevice);
        
        dim3 ts(32), bs(1344);
        add<<<bs, ts>>>(m, n, p, pitch_p, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c, dstds);  // 核函数
        div<<<bs, ts>>>(m, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c, dstds); 
        cudaDeviceSynchronize();

        if(m>0){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr0, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr0, corr_buf, sizeof(float)*m*M); 
        }
        if(m>1*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr1, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr1, corr_buf, sizeof(float)*m*M); 
        }
        if(m>2*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr2, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr2, corr_buf, sizeof(float)*m*M); 
        }
        if(m>3*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr3, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr3, corr_buf, sizeof(float)*m*M); 
        }
        if(m>4*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr4, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr4, corr_buf, sizeof(float)*m*M); 
        }
        if(m>5*M){
            cudaMemcpy2D(corr_buf, sizeof(float)*m, dcorr5, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
            memcpy(_corr5, corr_buf, sizeof(float)*m*M); 
        }

        cudaFree(p);                                                                                 
        cudaFree(dstds);
        cudaFreeHost(corr_buf);
        if(m>0)   cudaFree(dcorr0);                                            
        if(m>1*M) cudaFree(dcorr1);                                            
        if(m>2*M) cudaFree(dcorr2);                                            
        if(m>3*M) cudaFree(dcorr3);                                            
        if(m>4*M) cudaFree(dcorr4);                                            
        if(m>5*M) cudaFree(dcorr5);  

        return 0;
    }
}
