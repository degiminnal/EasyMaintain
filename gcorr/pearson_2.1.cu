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

#define M 9440
#define float float

#define get_id() (blockIdx.x * blockDim.x + threadIdx.x)
#define get_bid() (blockIdx.x)
#define get_tid() (threadIdx.x)


int m, n;                  // 基因表达数据的维度，m是基因数,基因数限制56640，n是基因的特征维度，即细胞数
float *a;                  // CPU存储的单细胞表达数据数组，维度为 m*n
float *p;                  // GPU存储的单细胞表达数据数组
float *dstds;              // GPU内存储基因的标准差
float *corr0;              // CPU内储存计算结果0
float *corr1;              // CPU内储存计算结果1
float *corr2;              // CPU内储存计算结果2
float *corr3;              // CPU内储存计算结果3
float *corr4;              // CPU内储存计算结果4
float *corr5;              // CPU内储存计算结果5
float *dcorr0;             // GPU内储存计算结果0
float *dcorr1;             // GPU内储存计算结果1
float *dcorr2;             // GPU内储存计算结果2
float *dcorr3;             // GPU内储存计算结果3
float *dcorr4;             // GPU内储存计算结果4
float *dcorr5;             // GPU内储存计算结果5
size_t pitch_p, pitch_c;   // GPU行间地址距离


void read_csv(string filename, string sep=",")
{
    ifstream csv_data(filename, ios::in);
    string line;

    if (!csv_data.is_open())
    {
        cout << "Error: opening file fail" << endl;
        exit(1);
    }

    istringstream sin;         
    vector<vector<float>> nums;
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
    a = (float*)malloc(sizeof(float)*m*n);
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

float* getptr(int r)
{
    int a=r/M, b=r%M;
    if(a==0) return corr0 + b * m;
    if(a==1) return corr1 + b * m;
    if(a==2) return corr2 + b * m;
    if(a==3) return corr3 + b * m;
    if(a==4) return corr4 + b * m;
    if(a==5) return corr5 + b * m;
    return nullptr;
}

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

int main(int argc, char **argv)
{
    string filename = "123.csv";
    string sep = ",";
    if (argc >= 2) filename = argv[1];
    if (argc >= 3) sep = argv[2];

    read_csv(filename, sep);

    float t1 = get_time();

    for(int i=0;i<m;i++){                                                          // 元素都减去其列均值，得到 delta 数组
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
    if(m>0)   cudaMallocHost((void**)&corr0, sizeof(float)*m*M);                   // CPU存储计算结果的内存0      
    if(m>1*M) cudaMallocHost((void**)&corr1, sizeof(float)*m*M);                   // CPU存储计算结果的内存1   
    if(m>2*M) cudaMallocHost((void**)&corr2, sizeof(float)*m*M);                   // CPU存储计算结果的内存2   
    if(m>3*M) cudaMallocHost((void**)&corr3, sizeof(float)*m*M);                   // CPU存储计算结果的内存3   
    if(m>4*M) cudaMallocHost((void**)&corr4, sizeof(float)*m*M);                   // CPU存储计算结果的内存4 
    if(m>5*M) cudaMallocHost((void**)&corr5, sizeof(float)*m*M);                   // CPU存储计算结果的内存4 
    if(m>0)   cudaMallocPitch((void**)&dcorr0, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存0 
    if(m>1*M) cudaMallocPitch((void**)&dcorr1, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存1   
    if(m>2*M) cudaMallocPitch((void**)&dcorr2, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存2 
    if(m>3*M) cudaMallocPitch((void**)&dcorr3, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存3 
    if(m>4*M) cudaMallocPitch((void**)&dcorr4, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存4    
    if(m>5*M) cudaMallocPitch((void**)&dcorr5, &pitch_c, sizeof(float)*m, M);      // GPU存储计算结果的内存5     

    cudaMemcpy2D(p, pitch_p, a, sizeof(float)*n, sizeof(float)*n, m, cudaMemcpyHostToDevice);
    
    dim3 ts(32), bs(1344);
    add<<<bs, ts>>>(m, n, p, pitch_p, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c, dstds);  // 计算皮尔逊相关系数的核函数
    div<<<bs, ts>>>(m, dcorr0, dcorr1, dcorr2, dcorr3, dcorr4, dcorr5, pitch_c, dstds); 
    cudaDeviceSynchronize();

    if(m>0)   cudaMemcpy2D(corr0, sizeof(float)*m, dcorr0, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
    if(m>1*M) cudaMemcpy2D(corr1, sizeof(float)*m, dcorr1, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
    if(m>2*M) cudaMemcpy2D(corr2, sizeof(float)*m, dcorr2, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
    if(m>3*M) cudaMemcpy2D(corr3, sizeof(float)*m, dcorr3, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
    if(m>4*M) cudaMemcpy2D(corr4, sizeof(float)*m, dcorr4, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);
    if(m>5*M) cudaMemcpy2D(corr5, sizeof(float)*m, dcorr5, pitch_c, sizeof(float)*m, M, cudaMemcpyDeviceToHost);



    t1 = get_time()-t1;
    cout<<"计算用时:"<<t1<<endl;


    for(int i=m-3;i<m;i++){
        for(int j=m-3;j<m;j++){
            cout<<getptr(i)[j]<<" ";
        }
        cout<<endl;
    }

    free(a);
    cudaFree(p);                                                                                 
    cudaFree(dstds);
    if(m>0)   cudaFree(dcorr0);                                            
    if(m>1*M) cudaFree(dcorr1);                                            
    if(m>2*M) cudaFree(dcorr2);                                            
    if(m>3*M) cudaFree(dcorr3);                                            
    if(m>4*M) cudaFree(dcorr4);                                            
    if(m>5*M) cudaFree(dcorr5);   
    if(m>0)   cudaFree(corr0);
    if(m>1*M) cudaFree(corr1);
    if(m>2*M) cudaFree(corr2);
    if(m>3*M) cudaFree(corr3);
    if(m>4*M) cudaFree(corr4);
    if(m>5*M) cudaFree(corr5);

    return 0;
}
