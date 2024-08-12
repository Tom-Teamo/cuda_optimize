#include <iostream>
#include <vector>

using namespace std;



// int main() {
//     // int n, m, k;
//     // cin >> n >> m >> k;
//     int n = 10, m = 4, k = 2;

//     int count = 0;
//     while (n != m) {
//         if (n % k == 0) {
//             if (n / k >= m) {
//                 n /= k;
//             }
//             else {
//                 n -= 1;
//             }
//         }
//         else {
//             n -= 1;
//         }
//         count ++;

//     }

//     return count;
// }


#include <string>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>

// s字符串最多能包含多少个a 和 b， s可以做重新排列和改变大小写

int main() {
    string s, a, b;
    cin >> s >> a >> b;

}

// 最多能添加多少条边 使得图依然是二分图

int main() {
    int n;
    cin >> n;
    unordered_map<int, unordered_set<int>> graph;
    for (int i = 0; i < n; i++)
    {
        int a, b;
        cin >> a >> b;
        graph[a].insert(b);
    }

    unordered_set<int> left, right;
    vector<int> remain;
    for (auto& [k, v] : graph)
    {
        if (left.find(k) == left.end() && right.find(k) == right.end())
        {
            if (left.size() ==0 && right.size() == 0) {
                left.insert(k);
                for (auto& i : v)
                {
                    right.insert(i);
                }
            }
            else {
                remain.push_back(k);
                continue;
            }
        }
        else if (left.find(k) != left.end())
        {
            for (auto& i : v)
            {
                right.insert(i);
            }
        }
        else if (right.find(k) != right.end())
        {
            for (auto& i : v)
            {
                left.insert(i);
            }
        }
    }

    int result = 0;
    for (int i : left) {
        for (int j: right) {
            if (graph[i].find(j) == graph[i].end() && graph[j].find(i) == graph[j].end()){
                result ++;
            }
        }
    }

   
    cout << result;
}
























// void mm(float* A, float* B, float* C, int M, int N, int K) {


//     for (int k = 0; k < K; k++)
//     {
//         for (int i = 0; i < M; i++)
//         {
//             for (int j = 0; j < N; j++)
//             {
//                 C[i * N + j] += A[i * K + k] * B[k * N + j];
//             }
//         }
//     }
    
// }


// // 4 5 6 7 8 0 1 2
// int half_half(vector<int> input, int target) {
//     int left = 0, right = input.size();
//     int mid = (right - left) / 2;

//     while (left <= right)
//     {
//         if (input.at(mid) > target) {
//             right = mid - 1;
//         }
//         else if (input.at(mid) < target)
//         {
//             left = mid + 1;
//         }
//         else {
//             return mid;
//         }
        
//     }
    

//     return -1;
// }


// int main() {
//     //  0 1 2 4 5 6 7 8
// }