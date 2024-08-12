#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <stack>
using namespace std;

/*
给出N个整数，可以随意挑选一些数字组成序列：
1. 奇偶交替
2. 前一个数小于后一个数

求出满足这两个要求的序列的最大长度
*/
// int main() {
//     int N;
//     cin >> N;
//     set<int> nums_set;
    

//     for (int i = 0; i < N; i++)
//     {   
//         int tmp;
//         cin >> tmp;
//         nums_set.insert(tmp);
//     }

//     vector<int> nums(nums_set.begin(), nums_set.end());

//     vector<int> dp(N, 1);
//     int res = 1;
//     for (int i = 1; i < N; i++)
//     {
//         for (int j = 0; j < i; j++)
//         {
//             if (nums[i] % 2 != nums[j] % 2 && nums[i] > nums[j]) {
//                 dp[i] = max(dp[i], dp[j] + 1);
//             }
//         }
//         res = max(res, dp[i]);
//     }

//     cout << res;

//     return 0;
// }

/*
有一组括号序列，包含[]{}四种括号
求最少修改多少次使得输入的序列是合法的
*/

int main() {
    int N;
    cin >> N;
    for (size_t i = 0; i < N; i++)
    {
        string s;
        cin >> s;
        stack<char> st;
        
        for (auto c: s) {
            char top = st.empty() ? ' ' : st.top();
            if (top == '[' && c == ']') {
                st.pop();
            }
            else if (top == '{' && c == '}') {
                st.pop();
            }
            else {
                st.push(c);
            }
        }
        int size = st.size();

        
        
        // vector<char> prev, next;
        // for (size_t i = 0; i < size / 2; i++)
        // {
        //     next.push_back(st.top());
        //     st.pop();
        // }
        // for (size_t i = 0; i < size / 2; i++)
        // {
        //     prev.push_back(st.top());
        //     st.pop();
        // }
        // reverse(next.begin(), next.end());
        
        // // prev: 3 2 1 0
        // // next：4 5 6 7
        // int res = 0;

        // for (size_t i = next.size() - 1; i > 0; i--)
        // {
        //     if (prev[i] == '[' && next[i] == ']') {
        //         continue;
        //     }
        //     else if (prev[i] == '{' && next[i] == '}') {
        //         continue;
        //     }
        //     else {
        //         res++;
        //     }
        // }


        
        // cout << res;
        
        // for (size_t i = 0; i < size / 2; i++)
        // {
        //     if (prev[i] == '[') {
                
        //     }
        //     else if (prev[i] == '{') {

        //     }
        //     else if (prev[i] == '}') {

        //     }
        //     else if (prev[i] == ']') {

        //     }
        // }
        

    }
    
    return 0;
}

