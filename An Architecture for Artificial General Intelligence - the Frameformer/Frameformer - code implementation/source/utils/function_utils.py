import time

def return_number_of_calls():
    return_number_of_calls.calls += 1
    return return_number_of_calls.calls

# Decorator: counts the number of calls to the llm, and stops the calls if llm is called more than 500 times in a single execution.
def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        if wrapper.calls > 500:
            raise Exception("Exceded 500 calls to the llm")
        print(f"Call {wrapper.calls} of {func.__name__}")
        return func(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        print(f"Execution time of {func.__name__}: {time.perf_counter_ns() - start} nanoseconds")
        return result

    return wrapper


#generates unique ids for objects based on the number this function is called.
def generate_id():
    generate_id.calls += 1
    return generate_id.calls