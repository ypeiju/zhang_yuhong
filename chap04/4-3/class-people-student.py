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
 
# 单继承示范

class Student(Person):
    grad = ''
    def __init__(this,name,age,weight,grad):
        #调用父类的构造方法，初始化父辈数据成员
        Person.__init__(this, name,age,weight)
        this.grade = grad

    #覆写父类的同名方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))


stu = Student('Alice',10,40,3)
stu.speak()


