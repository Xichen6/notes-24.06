# 欢迎使用Java
> 记录一些cs61b. 网站为[cs61b_sp21](https://sp21.datastructur.es/)
[TOC]

## 导读
- 入门课，前5周java，10周讲数据结构和算法，其中4节软件工程.
- Leetcode: 用某数据结构可以做到一些事情(例如用红黑树实现 log N 级别的插入). cs61B 倾向于解释为什么是 log N 级别的.

- 底部程序只有数组和链表两种形式. 所有数据结构都是数组和链表搭起来的
No data structures is magic!


## Lec 1. Introduction

- Java isn't the focus of the course!
- Writing code efficiently.
    - Designing, building, testing and debugging large programs.
- All code must be part of a class.
    - java has static types.

```java
public class Helloworld {
    public static void main(String[] args){
        System.out.println("hello world");
    }
}

public class HelloNumbers {
    public static void main(String[] args){
        int x;
        x = 0;
        while (x < 10){
            System.out.println(x);
            x ++;
        }

        String x; // can't change int --> string
        x = "horse";
        System.out.println(x);
    }
}

public class LargerDemo {
    public static int larger(int x, int y){
        if (x > y){
            return x;
        } else {
            return y;
        }
    }

    public static void main(String[] args){
        System.out.println(larger(-5, 10))
    }
}

public class Car {

    String model;
    int  wheels;

    public Car(String m) {  // Don't use __init__, but use Car
        this.model = m;
        this.wheels = 4;
    }

    public void drive(){ // void says nothing get return.
        if (this.wheels < 4){
            System.out.println(this.model + "no go vroom");
        }
        System.out.println(this.model + "go vroom");
    }

    public int getNumWheels(){
        return self.wheels;
    }

    public void driveIntoDitch(int wheelsLost){
        this.wheels = this.wheels - wheelsLost;
    }

    public static void main(String[] args){
        Car c1;
        Car c2;

        c1 = new Car("Civic Type R");
        c2 = new Car("Toyota Camry");

        c1.drive();
        c1.driveIntoDitch(2);
        c1.drive();

        System.out.println(c2.getNumWheels())

    }
}

// Discussion 1 - 1
public class Dog{
    int size;
    String name;
    public Dog(String n, int s){
        this.name = n;
        this.size = s;
    }

    public void bark(int x){
        x = this.size - 5;
    }

    public static void main(String[] args){
        Dog yourDog;
        List dogList;
        Dog dogList[3];

        yourDog = Dog("Scruffy", 1000);
        dogList[0] = myDog;
        dogList[1] = yourDog;
        doglist[2] = 5;
        dogList[3] = Dog("Cutie", 8)
        dogList = Dog[3]; // this is error

        if (x < 15) {
            myDog.bark(8);
            }

    // Dog myDog = new Dog(name, size);
    // Dog yourDog = new Dog("Scruffy", 1000);
    // Dog[] dogList = new Dog[3];
    // dogList[0] = myDog;
    // dogList[1] = yourDog;
    // doglist[2] = 5;
    // dogList[3] = new Dog("Cutie", 8)
    // int x;
    // x = size- 5;
    // if (x < 15) {
    // myDog.bark(8);
 }

}



```

```python
# python
class Car:
    def __init__(self, m):
        self.model = m
        self.wheels = 4

    def drive(self):
        if self.wheels < 4:
            print(self.model + "no go vroom")
        print(self.model + "goes vroom")

    def getNumWheels(self):
        return self.wheels

    def driveIntoDitch(self, wheelsLost):
        self.wheels -= wheelsLost

c1 = Car("Civic Type R")
c2 = Car("Toyota Camry")

c1.drive
c1.driveIntoDitch(2)
c1.drive

print(c2.getNumWheels())
```

- `while` Loop
Using `{}` to wrap the code that is part of the while loop. Java doesn't require **indenting**.
```java
// java
int i = 0;
while (i < 10){
    System.out.println(i);
    i ++;
}
```
```python
# python
i = 0
while i < 10:
    print(i)
    i += 1
```


- `for` Loop
```java
for (int i = 0; i < 10; i ++) {
    System.out.println(i)
}
```
 **Function Declaration**
- In java, functions have a specific **return** type that comes **before** the **funtion** name.
- When a function return nothing, it has a return type of **void**.

```java
public static String greet(String name) {
    return "Hello, " + name;
}

System.out.println(greet("Josh"))
/*
multi-line comments
*/
```

In java, Strings are not directly iterable.

```java
// 拼接 character 要先转换为 String, 但 character 拼接 String 不需要.
char c1 = 'a';
char c2 = 'b';
String result = c1.toString() + c2.toString();

String s = "hello";
s += "world";
s += 5;
int sLength = s.length();
String substr = s.substring(1, 5);
char c = s.charAt(2);

for (int i = 0; i < s.length(); i++) {
    char letter = s.charAt(i);
    System.out.println(letter);
}
s.charAt(2)
```

```python
# python
s = "hello"
s += " world"
s += str(5)
s_length = len(s)
substr = s[1:5]
c = s[2]
if "hello" in s:
    print("\"hello\" in s")

for letter in s:
    print(letter)
```

Java **arrays** are a lot like Python.
```java
// java



int[] array = {4, 7, 10};
array[0] = 5
System.out.println(array[0]);
System.out.println(Array.toString(array));
System.out.println(array.length);



```




```python
zeroedLst = [0, 0, 0]
lst = [4, 7, 10]
lst[0] = 5
print(lst[0])
print(lst)
print(len(lst))


lst = [1, 2, 3]
for i in lst:
    print(i)

```


```java
// java
public static int min_index() {
    
}


```





```python
# python
def min_index(numbers):
    # Assume len(numbers) >= 1
    m = numbers[0]
    idx = 0
    for i in range(len(numbers)):
        if numbers[i] < m:
            m = numbers[i]
            idx = i
    return idx
```




## Lec 2. Classes

```java
public class Dog{

    public int weightInPounds;

    public void makeNoise() {
        if (weightInPounds < 10) {
            System.out.println("yipyipyip!");
        } else if (weightInPounds < 30) {
            System.out.println("bark!");
        } else {
            System.out.println("arooooooo!");
        } 
    }

    public static void main(String[] args){
        makeNoise();
    }
}

public class DogLauncher{

        public static void main(String[] args){
            Dog d = new Dog();
            d.weightInPounds = 20;
            d.makeNoise();
        }
    
}






```


## Lec 12. Asymptostics 1
- 



# Lec 6. Version Control (git)



git uses directed cyclic graph to model history



Algorithm: well-defined procedure for carrying out some task.



## Leetcode 基础算法精讲

### 09 二叉树的最大深度 104


```python
# 时间、空间复杂度为 O(n)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

```

### 10 二叉树 相同 100 101 110 199



```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None or q is None:
            return p == q
        else:
            if p.val != q.val
                return False
            else:
                return self.isSameTree(self, p.left, q.left) and self.isSameTree(self, p.right, q.right)

```





### 17 动态规划 打家劫舍 198

基本思想是将 dp 问题转化为递归问题.
$$dfs(i) = max(dfs(i-1), dfs(i-2) + nums[i])$$
```python
# 递归写法, 空间复杂度为 O(1), 这是因为每次就存两个数
# 相比之下用 @cache 空间复杂度为 O(n)
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        f0 = f1 = 0
        for i, x in numerater(nums):
            f0, f1 = f1, max(f1, f0 + x)
        return f1


# 377. 组合总和
# 请注意这一段是不行的，因为 nums 是不可哈希的类型        
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int: 
        @cache
        def com(nums, target): # 改为 com(target)
            ans = 0
            # minum = min(nums)
            if target == 0:
                return 1
            elif target < 0:
                return 0
            for i in nums:
                # if target >= i:
                ans += com(nums, target - i)
            return ans
        return com(nums, target)

# 正确解答  
    def combinationSum4(self, nums: List[int], target: int) -> int: 
        zeros = [0] * (target + 1)
        zeros[0] = 1
        for i in range(target + 1):
            for j in nums:
                if i >= j:
                    zeros[i] += zeros[i - j]
        return zeros[-1]


# 215. 数组中的第K个最大元素
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 超出时间限制
        # for i in range(k):
        #     if k == 1:
        #         return max(nums)
        #     max_num = max(nums)
        #     nums.remove(max_num)
        #     k -= 1
```

