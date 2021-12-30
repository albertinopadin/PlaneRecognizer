from time import perf_counter


def timer(func):
    def wrapper(*args, **kwargs):
        start_t = perf_counter()
        r_val = func(*args, **kwargs)
        end_t = perf_counter()
        elapsed = end_t - start_t
        print(f"{func.__name__} took time: {elapsed} seconds, {elapsed/60} minutes")
        return r_val
    return wrapper
