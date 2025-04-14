from typing import Any

class D(object):
    def __init__(self, num1=0, num2=5):
        self.test=num1
        self.test2=num2
    def __getattribute__(self, __name: str) -> Any:
        if __name=='test':
            return 0.
        else:
            return super().__getattribute__(__name)
        

if __name__ == "__main__":
    d = D(10,20)
    print(d.test)
    print(d.test2)
   
   