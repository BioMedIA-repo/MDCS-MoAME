import h5py
import numpy as np
import os
from einops import rearrange, repeat
def left_oblique_scanning(x_axis= [], y_axis= []):
    result = []
    result.append([x_axis[0], y_axis[0]])
    x_len = len(x_axis)-1
    y_len = len(y_axis)-1
    i = 0
    j = 0
    while(1):
        if(i < x_len):
            i = i + 1
            result.append([x_axis[i], y_axis[j]])
            if (i == x_len and j == y_len):
                break
        elif(i == x_len and j < y_len):
            j = j + 1
            result.append([x_axis[i], y_axis[j]])
            if (i == x_len and j == y_len):
                break
        while(i > 0 and j < y_len):
            i = i - 1
            j = j + 1
            result.append([x_axis[i], y_axis[j]])
        if(j < y_len):
            j = j + 1
            result.append([x_axis[i], y_axis[j]])
            if (i == x_len and j == y_len):
                break
        elif(j == y_len and i < x_len):
            i = i + 1
            result.append([x_axis[i], y_axis[j]])
            if (i == x_len and j == y_len):
                break
        while(j > 0 and i < x_len):
            j = j - 1
            i = i + 1
            result.append([x_axis[i], y_axis[j]])
    result = np.array([p for p in result])
    return result

def right_oblique_scanning(x_axis= [], y_axis= []):
    result = []
    x_len = len(x_axis)-1
    y_len = len(y_axis)-1
    result.append([x_axis[x_len], y_axis[0]])
    i = x_len
    j = 0
    while(1):
        if(i > 0):
            i = i - 1
            result.append([x_axis[i], y_axis[j]])
            if (i == 0 and j == y_len):
                break
        elif(i == 0 and j < y_len):
            j = j + 1
            result.append([x_axis[i], y_axis[j]])
            if (i == 0 and j == y_len):
                break
        while(i < x_len and j < y_len):
            i = i + 1
            j = j + 1
            result.append([x_axis[i], y_axis[j]])
        if(j < y_len):
            j = j + 1
            result.append([x_axis[i], y_axis[j]])
            if (i == 0 and j == y_len):
                break
        elif(j == y_len and i > 0):
            i = i - 1
            result.append([x_axis[i], y_axis[j]])
            if (i == 0 and j == y_len):
                break
        while(j > 0 and i > 0):
            j = j - 1
            i = i - 1
            result.append([x_axis[i], y_axis[j]])
    result = np.array([p for p in result])
    return result

def loopback_scanning(x_axis= [], y_axis= []):
    result = []
    x_len = len(x_axis)-1
    y_len = len(y_axis)-1
    x_left = 0
    y_left = 0
    result.append([x_axis[0], y_axis[0]])
    i = 0
    j = 0
    index = int((x_len+1)*(y_len+1)-1)
    point = 0
    y_left = y_left + 1
    while(1):
        if(point == index):
            break
        while(i < x_len):
            i = i + 1
            result.append([x_axis[i], y_axis[j]])
            point = point + 1
        x_len = x_len - 1
        if (point == index):
            break
        while(j < y_len):
            j = j + 1
            result.append([x_axis[i], y_axis[j]])
            point = point + 1
        y_len = y_len - 1
        if(point == index):
            break
        while(i > x_left):
            i = i - 1
            result.append([x_axis[i], y_axis[j]])
            point = point + 1
        x_left = x_left + 1
        if(point == index):
            break
        while(j > y_left):
            j = j - 1
            result.append([x_axis[i], y_axis[j]])
            point = point + 1
        y_left = y_left + 1
        if(point == index):
            break
    result = np.array([p for p in result])
    return result

def compare_arrays(arr1, arr2):
    set_arr1 = {tuple(subarr) for subarr in arr1}
    set_arr2 = {tuple(subarr) for subarr in arr2}

    common_elements = set_arr1 & set_arr2
    common_elements = [subarr for subarr in arr2 if tuple(subarr) in common_elements]
    common_elements = np.array([p.tolist() for p in common_elements])
    return common_elements


folder_path = '/data/path/to/TCGA_BLCA/patch'
i = 1
for file_name in os.listdir(folder_path):
    if file_name.endswith('.h5'):
        file_path = os.path.join(folder_path, file_name)
    with h5py.File(file_path, 'a') as f:
        # print("Keys in the HDF5 file:", list(f.keys()))

        dataset = f['coords']
        data = dataset[:]

        # Vertical_scanning
        sorted_points = sorted(data, key=lambda p: (p[1], p[0]))
        points_list = np.array([p.tolist() for p in sorted_points])

        x_axis = np.array([p[0] for p in data])
        y_axis = np.array([p[1] for p in data])
        x_axis = sorted(set(x_axis))
        y_axis = sorted(set(y_axis))
        # Left_oblique_scanning
        result_topleft = left_oblique_scanning(x_axis=x_axis, y_axis=y_axis)
        result_topleft = compare_arrays(data, result_topleft)

        # Right_oblique_scanning
        result_topright = right_oblique_scanning(x_axis=x_axis, y_axis=y_axis)
        result_topright = compare_arrays(data, result_topright)

        # Loopback_scanning
        result_hui = loopback_scanning(x_axis=x_axis, y_axis=y_axis)
        result_hui = compare_arrays(data, result_hui)

        if 'coords_vet' not in f:
            f.create_dataset('coords_vet', data=points_list)
        if 'topleft' not in f:
            f.create_dataset('topleft', data=result_topleft)
        if 'topright' not in f:
            f.create_dataset('topright', data=result_topright)
        if 'hui' not in f:
            f.create_dataset('hui', data=result_hui)
            print('Processed the {}th file'.format(i))
            i = i + 1
