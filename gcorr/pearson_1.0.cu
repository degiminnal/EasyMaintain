
#include <cuda.h>
#include <iostream>
#include <windows.h>

#include <istream>
#include <streambuf>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>


using namespace std;

#define get_id() (blockIdx.x * blockDim.x + threadIdx.x)
#define get_bid() (blockIdx.x)
#define get_tid() (threadIdx.x)

#define N 10000                                              // 单个基因的最大特征维度，即细胞样本数


int m, n;                                                    // 基因表达数据的维度，m是基因数，n是基因的特征维度，即细胞数
float (*p)[N];                                               // 数组指针，最多六万个数组（每个数组最多10000个元素，可以扩充），即整个数组最多 6e8g 个单精度浮点数
float *dx, *dy, *dz;                                         // 两个向量dx,dy，以及向量对应元素乘积和部分规约后的中间结果
float *corr[30000];                                          // 定义结果数组，表示基因对之间的关系，限制最多三万个基因，可以扩充
float *stds;                                                 // 存储基因的标准差
float *a;


void read_csv(string filename)
{
    ifstream csv_data(filename, ios::in);
    string line;

    if (!csv_data.is_open())
    {
        cout << "Error: opening file fail" << endl;
        exit(1);
    }

    istringstream sin;         
    vector<vector<float>> nums; //声明一个字符串向量
    string word;

    m = 0;
    getline(csv_data, line);
    for(int i=0, j=0; i<line.size(); i++){
        if(line[i]==','){
            string tmpstr = line.substr(j, i-j);
            if(tmpstr.size()<=2 && tmpstr[0]=='y') break;
            j = i + 1;
            m++;
        }
    }
    // cout<<m<<endl;
    // 读取数据
    m = 10;
    while (getline(csv_data, line))
    {
        sin.clear();
        sin.str(line);
        nums.push_back(vector<float>{});
        for(int i=0;i<m;i++){
            getline(sin, word, ',');
            nums.back().push_back(stof(word));
        }
    }
    csv_data.close();
    n = nums.size();
    a = (float*)malloc(4*m*n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            a[i*n+j] = nums[j][i];
        }
    }
}

float get_time(void)
{
    LARGE_INTEGER timer;
    static LARGE_INTEGER fre;
    static int init = 0;
    double t;

    if (init != 1) {
        QueryPerformanceFrequency(&fre);
        init = 1;
    }

    QueryPerformanceCounter(&timer);

    t = timer.QuadPart * 1. / fre.QuadPart;

    return t;
}



__device__ void wrapReduction(volatile float* sdata, int tid)
{
    for(int t=32;t>=1;t/=2){
        sdata[tid] += sdata[tid+t];
    }
}

__global__ void add(float* x, float* y, float* z, int n)
{
    __shared__ float sdata[256];
    int idx = get_id();
    int tid = get_tid(), bid = get_bid();
    if(idx<n){
        sdata[tid] = x[idx] * y[idx];
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    for(int t=128;t>32;t/=2){
        if(tid<t) sdata[tid] += sdata[tid+t];
        __syncthreads();
    }

    if(tid<32) wrapReduction(sdata, tid);

    if(tid==0) z[bid] = sdata[0];
}

__global__ void reduction(float* x, int n)
{
    __shared__ float sdata[256];
    int tid = get_tid();
    sdata[tid] = 0;
    for(int idx=tid; idx<n; idx+=256){
        sdata[tid] += x[idx];
    }
    
    __syncthreads();

    for(int t=128;t>32;t/=2){
        if(tid<t) sdata[tid] += sdata[tid+t];
        __syncthreads();
    }

    if(tid<32) wrapReduction(sdata, tid);

    if(tid==0) x[0] = sdata[0];
}

void calc(int x, int y)
{
    dx = (float *)(p + x);
    dy = (float *)(p + y);
    dim3 ts(256), bs((n+255)/256);
    add<<<bs, ts>>>(dx, dy, dz, n);
    reduction<<<1, 256>>>(dz, bs.x);
    cudaMemcpy((void *)&corr[x][y], (void *)(dz), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&corr[y][x], (void *)(dz), sizeof(float), cudaMemcpyDeviceToHost);
}


int main()
{
    //float a[18]={1,1,1,1,1,1,3,8,9,13,11,15,8,7,6,5,4,1};    // 原始数据输入展平成一维的原始数据，按列展开
    read_csv("lake.csv");

    float t1 = get_time();

    for(int i=0;i<m;i++){                                    // 每一列的元素都减去其均值，得到 delta 数组
        float sum = 0;
        for(int j=0;j<n;j++){
            sum += a[i*n+j];
        }
        float mean = sum / n;
        for(int j=0;j<n;j++){
            a[i*n+j] -= mean;
        }
    }

    for(int i=0;i<m;i++){
        cudaMallocHost((void**)&corr[i], 4*m);               // 为结果数组分配内存
        // corr[i] = (float *)malloc(4*m);
    }                                                 
    cudaMalloc((void**)&p, m*N*4);                           // 在 GPU 内部分配 delta 数组空间
    cudaMalloc((void**)&dz, N*4);                            // 存储dx，dy临时乘积的数组   
    for(int i=0;i<m;i++){                                    // 拷贝 delta 数组到 GPU 内存
        cudaMemcpy((void *)(p+i), (void *)(a+i*n), 4*n, cudaMemcpyHostToDevice);
    } 
    stds = (float*)malloc(m*sizeof(float));

    for(int i=0;i<m;i++){
        for(int j=i;j<m;j++){
            calc(i, j);
        }
    }

    t1 = get_time()-t1;
    cout<<"计算用时:"<<t1<<endl;

    for(int i=0;i<m;i++) stds[i] = sqrt(corr[i][i]);
    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            corr[i][j] /= stds[i]*stds[j];
        }
    }
    cudaFree(p);                                             // 释放内存
    cudaFree(dz);
    for(int i=0;i<m;i++){
        cudaFreeHost(corr[i]);
        // free(corr[i]);
    }
    free(stds);

    return 0;
}
