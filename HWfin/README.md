# HWfin: Code Clone Detection

    Homework code for course Intro: Data Mining

----

# [!!]想要复现结果只需要

    - 运行 `make ts`  或者 `python clone_detect.py ts`，大约耗时3min
    - 输出结果文件在 `dataset/result.csv`

### requirements

  - `pip install sqlalchemy pycparser zss`
  - `pip install numpy pandas`
  - `pip install sklearn keras`

## Quick Start: these steps must executed sequentially

  - `make rv`: prebuild all reprvec (51500 = 5.15w)
  - `make fv_ts`: prebuild all test featvec (200000 = 20w)
  - `make fv_tr N=0`: prebuild random train featvec of given size
  - `make tr N=0`: train & save model using random train featvec of given size (0 for all cached)
  - `make ts N=0`: load model, predict on test, write result csv
  - `make cv`: show CodeViewr(GUI) for manual check or for intuition

```
| Time Cost     | no TD-tree | with TD-tree   | 
| :-: | :-: | :-: |
| make rv       |    ?       |   15min  |
| make fv_ts    | 20min      |     ?    |
| make fv_tr    | 40min(40w) |  ? (30w) |
| make tr       | 30min      |   30min  |
| make ts       |  3min      |   2min   |
| <FINAL SCORE> | 0.46516    |  0.51248 |
```

### Feature Engineering Intuition

相似功能/语义的代码应该有：

  - 相同的常量值
    - 区分一般字面值和数组大小(地址值)?
      - 数值型常量用作数组大小时不一定是10的幂，考虑舍入?
    - 分离格式化字符串的常见前后缀：'\n', ','
    - 合并语义相同的常量：如endl和'\n'
  - 相同的变量名(忽略大小写)
  - 相近的变量数量
  - 相近的变量类型的数量分布
    - 字符型: char
    - 文本型: string/char[]
    - 整型: short/int/long/signed/unsigned
    - 浮点型: float/double
    - 数组: []/[][]/[][][]
    - 结构体
  - 相近的控制结构数量
    - 条件分支: if/case
    - 循环体块: for/while
  - 相近的库函数调用
    - IO
      - cin/Xscanf/getX/getline
      - cout/Xprintf/putX
    - MEMORY
      - malloc/free
    - string.h
      - strX
      - memX
    - math.h
      - sqrt/logX/exp
      - floor/ceil/sin/cos
  - 相似的AST树
    - 较小的编辑距离
    - 相近的树高度

对每份源代码产生一个向量表示reprvec:

  - 具体特征参见`Visitor.abstract_reprvec()`

对每两个reprvec产生一个用作训练或测试的featvec:

  - 具体特征参见`abstract_featvec()`

### references

 - [代码克隆检测研究进展](http://www.jos.org.cn/html/2019/4/5711.htm)
 - [Tree Edit Dist](https://github.com/timtadh/zhang-shasha)

----

by Armit
2020/06/07 
