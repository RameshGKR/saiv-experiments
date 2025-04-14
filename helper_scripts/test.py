from typing import Any

class Test_Set():
    def __init__(self, in_list=[], out_list=[]):
        if not isinstance(in_list, list) or not isinstance(out_list, list):
            raise AttributeError("in_list and out_list have to be lists")
        if (len(out_list) != 0) and (len(in_list) != 0) and  (len(out_list) != len(in_list)):
            raise AttributeError("when the input and output are not empty the length of input and output should match")
        
        self.input_list = in_list
        self.output_list = out_list

        self.length = max(len(in_list),len(out_list))

    def __give_input__(self):
        input=Test_Set(in_list=self.input_list)
        return input

    def __give_output__(self):
        output=Test_Set(out_list=self.output_list)
        return output
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "input":
            if not isinstance(__value, list):
                raise AttributeError("input has to be a list")
            if len(self.output_list) != 0 and  len(self.output_list) != len(__value):
                raise AttributeError("when the output is not empty the length of input should match")
            
            self.length = len(__value)
            super().__setattr__(self.input_list, __value)

        elif __name == "output":
            if not isinstance(__value, list):
                raise AttributeError("output has to be a list")
            if len(self.input_list) != 0 and  len(self.input_list) != len(__value):
                raise AttributeError("when the input is not empty the length of input should match")
            
            self.length = len(__value)
            super().__setattr__(self.output_list, __value)
        else:
            super().__setattr__(__name, __value)

    def __getattribute__(self, __name: str) -> Any:
        if __name == "input":
            return self.__give_input__()
        elif __name == "output":
            return self.__give_output__()
        else:
            return super(Test_Set, self).__getattribute__(__name)
        
if __name__ == "__main__":
    test_set = Test_Set(in_list=[1,2,3], out_list=[4,5,6])
    print(test_set.input_list)
    p=test_set.input
    b=test_set.output
    b.output_list=[7,8,9]
    print('hey')
    
   