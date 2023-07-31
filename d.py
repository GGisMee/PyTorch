class MyClass:
    def __init__(self, *args):
        # You can handle the input arguments as needed
        self.variables = args



def create_object_and_initialize_class(my_class, another_variable="default_value", *args):
    # Create the class object and pass the arbitrary number of variables to initialize it
    obj = my_class(*args)
    return obj, another_variable



obj, constant_variable = create_object_and_initialize_class(MyClass, 42, "Hello", [1, 2, 3], {"key": "value"})
print(obj.variables)  # Output: (42, 'Hello', [1, 2, 3], {'key': 'value'})
print(constant_variable)  # Output: 42

obj2, constant_variable2 = create_object_and_initialize_class(MyClass, "Hello", [1, 2, 3], {"key": "value"})
print(obj2.variables)  # Output: ('Hello', [1, 2, 3], {'key': 'value'})
print(constant_variable2)  # Output: default_value
