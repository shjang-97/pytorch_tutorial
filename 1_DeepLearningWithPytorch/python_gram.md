# Class 상속 및 super 함수의 역할
## 요약
### super() 관련 표현
class name(ClassName)를 상속 받을 때
- python 2.x 문법  

```
super(ClassName, self).__init__()
super(ClassName, self).__init__(**kwargs)
```

- python 3.x 문법
```
super().__init__()
super().__init__(**kwargs)
```
참고로 파이썬3 버전은 2버전의 문법도 함께 사용이 가능  
코드의 범용성을 위해 파이썬2 버전의 문법으로 적어주는 것이 좋다  

### super(): 부모 클래스
- 코드
```buildoutcfg
class Parent:
    def __init__(self, p1, p2):
    	'''super()를 사용하지 않으면 overriding 됩니다.'''
        self.p1 = p1
        self.p2 = p2
        
class Child(Parent):
    def __init__(self, c1, **kwargs):
        super(Child, self).__init__(**kwargs)
        self.c1 = c1
        self.c2 = "This is Child's c2"
        self.c3 = "This is Child's c3"

child = Child(p1="This is Parent's p1", 
	          p2="This is Parent's p1", 
              c1="This is Child's c1")

print(child.p1)
print(child.p2)
print(child.c1)
print(child.c2)
print(child.c3)
```
- 결과
```buildoutcfg
This is Parent's p1
This is Parent's p1
This is Child's c1
This is Child's c2
This is Child's c3
```
#### **코드가 실행되는 이유?**
1. 자식 클래스는 부모 클래스를 상속받는다. (자식 클래스가 부모 클래스의 메소드를 모두 가져온다고 생각하면 쉬움)  
1. 그러나 자식과 부모 클래스 모두 `__init()__`이라는 메소드를 가지고 있음.  
따라서 `super()`을 하지 않으면 부모 클래스의 `__init__()`은 자식 클래스의 `__init__()`에 의해 덮어쓰기 됨
1. 자식 클래스의 `__init__()` 메소드에 부모 클래스의 `__init__()` 메소드의 변수를 가지고 오고 싶을 때, 자식 클래스의 `__init__()` 메소드에서 `super().__init__()`을 입력하면 부모 클래스의 `__init__()`에 있는 클래스의 변수를 가지고 올 수 있음
1. 즉, `super()`은 상속받은 부모 클래스를 의미
1. 그러나 부모 클래스의 `__init__()` 메소드에 *argument*가 있다면 자식 클래스는 `__init__()`에 ***kwargs*를 반드시 입력해줘야 함  
(하지 않을 경우 *__init__() missing required positional arguments* 에러가 발생)

-출처: <https://velog.io/@gwkoo/%ED%81%B4%EB%9E%98%EC%8A%A4-%EC%83%81%EC%86%8D-%EB%B0%8F-super-%ED%95%A8%EC%88%98%EC%9D%98-%EC%97%AD%ED%95%A0>