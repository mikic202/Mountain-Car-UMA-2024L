import numpy as np

def get_discretized_array(min_value, max_value, step, precision=3):
    if min_value > max_value:
        raise ValueError("min_value must be less than max_value")
    available_values = []
    current_value = min_value
    while current_value < max_value:
        available_values.append(current_value)
        current_value += step
        current_value = round(current_value, precision)
    
    if round(current_value - step - max_value,3) != 0:
        available_values.append(max_value)
    return available_values


def get_discretized_array_numpy(min_value, max_value, step, precision=3):
    if min_value > max_value:
        raise ValueError("min_value must be less than max_value")
    available_values = []
    current_value = min_value
    while current_value < max_value:
        available_values.append(np.array([current_value], dtype=np.float32))
        current_value += step
        current_value = round(current_value, precision)
    
    if round(current_value - step - max_value,3) != 0:
        available_values.append(np.array([max_value], dtype=np.float32))
    return available_values


def find_closest(array, number):
    closest_number = min(array, key=lambda x: abs(x - number))
    return closest_number