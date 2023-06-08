import numpy as np
import math
import json
import time

# Author: oLotuSo
# Date: 2023-4-19

#=======超参数设定=======
filename = "./WikiData.txt"
output_path = './out2.txt'
parameter_r_path = './parameter_r.txt'
parameter_M_path = './parameter_M.txt' #分别在硬盘上存储r和M参数,模拟大数据下分块读入写出的过程
list = []
Beta = 0.85
N = 8297 #图中结点个数
block_size = 128 #读入内存块数
stripe = math.ceil(N/block_size) #分条个数
#=======================

def Data_to_Graph(G): #构造邻接矩阵
    with open(filename) as f:
        for line in f.readlines():
            list = line.split()
            i = int(list[0]) - 1
            j = int(list[1]) - 1
            G[i][j] = 1 
    row = np.sum(G, axis=1)
    col = np.sum(G, axis=0)
    cnt = 0 
    for i in range(N):
        if int(row[i] + col[i]) != 0: #剔除无关节点,在图上表现为去除孤立点,它们对PR值没有任何贡献
            cnt += 1
    return cnt, G

def Graph_to_Matrix(G): #构造随机邻接矩阵
    M = np.zeros((N, N))
    for i in range(N):
        D_i = sum(G[i]) #计算出度
        if D_i == 0:
            continue
        for j in range(N):
            M[j][i] = G[i][j] / D_i #注意这里是Mji
    return M

def Sparse_Matrix(G): # 构造(src, degree, dest)来表示稀疏矩阵,并分条
    encode = [[] for i in range(stripe)]
    for i in range(N):
        D_i = int(sum(G[i]))
        if D_i != 0:
            for j in range(stripe):
                dest2 = []
                for k in range(j * block_size, min((j + 1) * block_size, N)):
                    if G[i][k] != 0:
                        dest2.append(k)
                if dest2: encode[j].append([i, D_i, dest2])
    with open(parameter_M_path, 'w') as f:
        f.write(json.dumps(encode))  #利用json保存list数组

def Block_Stripe_Update(cnt = N, eps = 1e-10, beta = Beta): # 分块条形更新算法
    R = (1 - beta) * np.ones(N) / cnt  #除以cnt是去除了无关点,如不去除无关点这里是N
    R_new = R.copy()
    np.savetxt(parameter_r_path, R)
    with open(parameter_M_path) as f:
        encode = json.loads(f.read())  
    while True:
        R = np.loadtxt(parameter_r_path) # 每次迭代前先读取R,模拟大数据下分块读入的过程
        for i in range(stripe):
            for j in range(len(encode[i])):
                for k in range(len(encode[i][j][2])):
                    R_new[encode[i][j][2][k]] += beta * R[encode[i][j][0]] / encode[i][j][1]
        if np.linalg.norm(R_new - R) < eps: #收敛
            break    
        np.savetxt(parameter_r_path, R_new) # 每次迭代后保存R,模拟大数据下分块写出的过程
        R_new = (1 - beta) * np.ones(N) / cnt
    return R_new

def PageRank_Google(M, cnt = N, eps = 1e-10, beta = Beta): # Pagerank算法
    R = np.ones(N) / cnt
    teleport = np.ones(N) / cnt #除以cnt是去除了无关点,如不去除无关点这里是N
    while True:
        R_new = beta * np.dot(M, R) + (1 - beta) * teleport #考虑dead end 和 spider trap,不断迭代
        if np.linalg.norm(R_new - R) < eps: #收敛
            break
        R = R_new.copy()
    return R_new

def main():
    start_time = time.time()
    G = np.zeros((N, N))
    print("正在导入数据...")
    cnt, G = Data_to_Graph(G)
    print("优化稀疏矩阵...")
    Sparse_Matrix(G)
    print("计算Pagerank值...")
    R = Block_Stripe_Update(beta = Beta) #采用分块条形更新方法
    # M = Graph_to_Matrix(G) 
    # R = PageRank_Google(M, cnt, beta = Beta) #采用传统方法
    result = sorted(enumerate(R), key=lambda R:R[1], reverse = True)  #(NodeId, Score)元组排序
    with open(output_path, 'w') as file1:
        for i in range(100):
            NodeId, Score = result[i]
            print("[{}] [{}]".format(NodeId + 1, Score), file = file1) 
    end_time = time.time()
    print("time cost:", end_time - start_time, "s")
    return 0

if __name__ == "__main__":
    main()