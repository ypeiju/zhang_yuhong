class Person:
    height = 140    #定义类的数据成员
    #定义构造方法
    def __init__(self,name,age,weight):
        self.name = name    #定义对象的数据成员属性
        self.age = age
        #定义私有属性,私有属性在类外部无法直接进行访问
        self.__weight = weight
    def speak(self):
        print("%s 说: 我 %d 岁，我体重为 %d Kg，身高为 %d cm" %(self.name,self.age,self.__weight, Person.height))
 
# 实例化类
p1 = Person ('Alice',9,30)     # 实例化people类
p1.speak()                #引用对象中的公有方法
p1.age = 10
p1.name = 'Bob'
p1.speak()
p2 =  Person ('Luna',10,31)
Person.height = 150
p1.speak()
p2.speak()


