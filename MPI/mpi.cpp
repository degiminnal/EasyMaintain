#include <mpi.h>
#include<iomanip>
#include<iostream>
#include<sys/time.h>
#include<unordered_map>
#include"IterSuper.cpp"

#define Max_Matrix_size 100
using namespace std;

unordered_map<string, double> unmap;


double calc(Args args, double** matrix, int size)
{
	int n = args.s1.size();
	int a = 0, b = -1, k=1;
	double res = 0;
	while (args.s1[a] == '0') a += 1;
	args.s1[a] = '0';
	while (++b < n) {
		if (args.s2[b] == '0') continue;
		args.s2[b] = '0';
		res += (double)k *  matrix[a][b] * unmap[args.s1 + args.s2];
		k *= -1;
		args.s2[b] = '1';
	}
	return res;
}

double calc_matrix(double **matrix, int size)
{
    Iter iter = Iter(size);
	unmap.clear();
	unmap.emplace(string(size * 2, '0'), 1);
	Args args = iter.next();
	while (args.n)
	{
		unmap.emplace(args.s1+args.s2, calc(args, matrix, size));
		args = iter.next();
	}
	return unmap[string(size * 2, '1')];
} 

int main(int argc,char *argv[])
{
	MPI_Status status;
	double ans = 0.0, tmp;
	int matrix_size, process_num, myid;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &process_num);
	if (process_num < 2){
		cout<<"process number must be greater than 1"<<endl;
		return 0;
	} 
	if(argc != 3){
		cout<<"Input format:matrix_size  data_path"<<endl;
		return 0;
	}
    matrix_size = atoi(argv[1]);
    char* input = argv[2];
	double matrix[Max_Matrix_size][Max_Matrix_size];
	if (myid == 0) {
		char* input = "numbers.txt";
		freopen(input,"r",stdin);
		for(int i = 0;i < matrix_size;i++){
			for(int j = 0;j < matrix_size;j++){
				cin>>matrix[i][j];
			}
		}
	}
	for(int i=0;i<matrix_size;i++) 
		MPI_Bcast(matrix[i], matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	struct timeval begin,end;
	gettimeofday(&begin,NULL);
	int cnt = matrix_size / (process_num - 1) + (matrix_size % (process_num - 1) > 0);
	for(int i=0;i<process_num-1;i++){
		for(int j=0; j<cnt && i*cnt+j<matrix_size; j++){
			int idx = i * cnt + j;
			if (myid == i + 1) {
				double res = idx % 2 ? 1 : -1;
				double** mmatrix = (double**)malloc(sizeof(double*) * (matrix_size-1));
				for (int i = 0; i < matrix_size - 1; i++) {
					mmatrix[i] = (double*)malloc(sizeof(double) * (matrix_size - 1));
				}
				for (int j = 1; j < matrix_size; j++) {
					for (int k = 0; k < matrix_size; k++) {
						if (k < idx) mmatrix[j - 1][k] = matrix[j][k];
						else if(k > idx) mmatrix[j - 1][k - 1] = matrix[j][k];
					}
				}
				res *= calc_matrix(mmatrix, matrix_size - 1);
				MPI_Send(&res, 1, MPI_DOUBLE, 0, idx, MPI_COMM_WORLD);
				for (int i = 0; i < matrix_size - 1; i++) free(mmatrix[i]);
				free(mmatrix);
			}
			if (myid == 0) {
				MPI_Recv(&tmp, 1, MPI_DOUBLE, i+1, idx, MPI_COMM_WORLD, &status);
				ans += tmp;
			}
		}
	}
	if (myid == 0) {
		cout<<fixed<<setprecision(2)<<ans<< "\t";
		gettimeofday(&end,NULL);
		int time = (end.tv_sec - begin.tv_sec) * 1000000 + (end.tv_usec - begin.tv_usec);
		cout<<fixed<<setprecision(3)<<(double)time/1e6<<"s"<<endl;
	}
	MPI_Finalize();
	return 0;
}
