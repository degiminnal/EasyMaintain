#include<iostream>
using namespace std;

struct Args {
    int n = 0;
    string s1;
    string s2;
};

class Iter
{
        int MAX_SIZE;
        int cur_size;
        long long cur_max_idx;
        long long x_idx;
        string s_x, s_y;
        void get_cmi()
        {
            int n = cur_size;
            long long s1 = 1, s2 = 1;
            for(int i=0;i<n;++i){
                s1 *= (i+1);
                s2 *= (MAX_SIZE-i);
            }
            cur_max_idx = s2 / s1;
        }
        void nxt_s(string& s)
        {
            int i = s.size()-1;
            while(s[i]=='1') i-=1;
            int a = i;
            while(s[i]=='0') i-= 1;
            int b = i;
            s[i++] = '0';
            while(i<s.size()){
                if(i-b<=s.size()-a) s[i]='1';
                else s[i]='0';
                i += 1;
            }
        }
    public:
        Iter(int n)
        {
            MAX_SIZE = n;
            cur_size = 0;
            cur_max_idx = 0;
            x_idx = 0;
            s_x = string(n, '0');
            s_y = string(n, '0');
        }

        Args next()
        {
            Args arg;
            if(x_idx==cur_max_idx){
                if(cur_size==MAX_SIZE) return arg;
                cur_size += 1;
                get_cmi();
                x_idx =  1;
                s_x = string(cur_size, '1') + string(MAX_SIZE-cur_size, '0');
                s_y = string(MAX_SIZE - cur_size, '0') + string(cur_size, '1');
            }
            else{
                x_idx += 1;
                nxt_s(s_x);
            }
            arg.s1 = s_y;
            arg.s2 = s_x;
            arg.n = cur_size;
            return arg;
        }
};

