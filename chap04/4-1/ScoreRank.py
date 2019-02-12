#这是一个演示while循环的范例
while True:
    score = int(input("Please input your score : "))
    if 90 <= score <= 100:
        print('A')
    elif score >= 80:
        print('B')
    elif score >= 70:
        print('C')
    elif score >= 60:
        print("D")
    else:
        print('''你的分数有点低！''')
