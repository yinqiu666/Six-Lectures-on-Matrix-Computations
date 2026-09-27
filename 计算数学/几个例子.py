import numpy as np
import sympy as sy
#1.解方程
x=sy.symbols("x")
equation=sy.Eq(x**2-10000000000+1,0)
solution=sy.solve(equation,x)
print(solution)
#2.求积分
n=sy.symbols("n")
integral_expr = sy.integrate(x**n * sy.exp(x), x)
print(f"indefinite integral: {integral_expr}")
definite = sy.integrate(x**n * sy.exp(x), (x, 0, 1))
print(f"definite integral: {definite}")
#3.解线性方程组
# 定义系数矩阵和常数向量
A = sy.Matrix([[2, 3], [1, -1]])
b = sy.Matrix([8, 1])

# 求解
x = A.solve(b)
print("the exact solution:")
sy.pprint(x)
#计算特征值

# 定义矩阵
A = sy.Matrix([[4, -2],
            [1,  1]])

# 4.
# 计算特征值和特征向量
eigenvalues = A.eigenvals()  # 返回 {特征值: 重数}
eigenvectors = A.eigenvects()  # 返回 [(特征值, 重数, [特征向量列表])]

print("eigenvalues:")
sy.pprint(eigenvalues)
print("\neigenvectors:")
sy.pprint(eigenvectors)

print("\n" + "="*50)