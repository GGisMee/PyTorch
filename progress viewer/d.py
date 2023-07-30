class OuterClass:
    def __init__(self, outer_attr):
        self.outer_attr = outer_attr

    def outer_method(self):
        print("This is the outer method.")

    class InnerClass:
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance

        def inner_method(self):
            print("This is the inner method.")
            print("Accessing outer attribute:", self.outer_instance.outer_attr)
            self.outer_instance.outer_method()


# Creating an instance of the OuterClass
outer_instance = OuterClass(42)

# Creating an instance of the InnerClass within the OuterClass and passing the outer instance explicitly
inner_instance = OuterClass.InnerClass(outer_instance)

# Calling the inner method
inner_instance.inner_method()
