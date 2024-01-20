#include <bits/stdc++.h> 
void arraySortedOrNot(vector<int> &nums, int n)
{
    // Array has one or no element or the
    // rest are already checked and approved.
    if (n == 1 || n == 0)
        return Yes;
 
    // Unsorted pair found (Equal values allowed)
    if (nums[n - 1] < nums[n - 2])
        return No;
 
    // Last pair was sorted
    // Keep on checking
    return arraySortedOrNot(nums, n - 1);
}
vector<int> separateNegativeAndPositive(vector<int> &nums){
    int i=0;
    for(i=0;i<nums.size();i++){
        int minIndex=i;
        for(int j=i+1; j<nums.size(); j++){
            if(nums[i]>nums[minIndex]){
                minIndex=j;

            }
            swap(nums[minIndex], nums[i]);
        }

    }
    arraySortedOrNot(nums,nums.size());

    
}
