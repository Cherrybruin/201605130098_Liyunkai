# coding=utf-8
###################################

# 请根据需求自己补充头文件、函数体输入参数。

import torch
import numpy

###################################
# 2 Vectorization
###################################

torch.manual_seed(0)
numpy.random.seed(0)


def vectorize_sumproducts(array1, array2):
    """
     Takes two 1-dimensional arrays and sums the products of all the pairs.
    :return:
    """
    return torch.sum(torch.from_numpy(array1) * torch.from_numpy(array2))


def vectorize_Relu(array):
    """
    Takes one 2-dimensional array and apply the relu function on all the values of the array.
    :return:
    """
    return torch.relu(torch.from_numpy(array)).numpy()


def vectorize_PrimeRelu(array):
    """
    Takes one 2-dimensional array and apply the derivative of relu function on all the values of the array.
    :return:
    """
    temp_cal = torch.nn.Parameter(torch.from_numpy(array).float())
    relu_result = torch.sum(torch.relu(temp_cal))
    relu_result.backward()
    return temp_cal.grad.numpy()


######################################
# 3 Variable length
######################################

# Slice

def slice_fixed_point(array, start_pos, length):
    """
    Takes one 3-dimensional array with the starting position and the length of the output instances.
    Your task is to slice the instances from the same starting position for the given length.
    :return:
    """
    start_x, start_y, start_z = start_pos
    x_length, y_length, z_length = length
    return [
        [
            dim2[start_z:start_z + z_length]
            for dim2 in dim1[start_y:start_y + y_length]
        ] for dim1 in array[start_x:start_x + x_length]
    ]


def slice_last_point(array, l):
    """
     Takes one 3-dimensional array with the length of the output instances.
     Your task is to keeping only the l last points for each instances in the dataset.
    :return:
    """
    return [
        [
            dim2[-l:]
            for dim2 in dim1
        ]
        for dim1 in array
    ]


def slice_random_point(array, l):
    """
     Takes one 3-dimensional  array  with  the  length  of the output instances.
     Your task is to slice the instances from a random point in each of the utterances with the given length.
     Please use function numpy.random.randint for generating the starting position.
    :return:
    """
    x_length, y_length = l
    length = array.shape[2]
    result = numpy.empty((array.shape[0], x_length, y_length,), dtype=numpy.float)
    for i in range(array.shape[0]):
        start_x, start_y = numpy.random.randint(array.shape[1] - x_length), numpy.random.randint(
            array.shape[2] - y_length)
        result[i, :, :] = array[i, start_x:start_x + x_length, start_y:start_y + y_length]
    return result


# Padding

def pad_pattern_end(array):
    """
    Takes one 3-dimensional array.
    Your task is to pad the instances from the end position as shown in the example below.
    That is, you need to pad the reflection of the utterance mirrored along the edge of the array.
    :return:
    """
    max_batch = (max([batch.shape[0] for batch in array]), max([batch.shape[1] for batch in array]))
    for batch in array:
        batch = numpy.pad(batch, ((0, max_batch[0]), (0, max_batch[1])), "reflect")
    return array


def pad_constant_central(array, acval):
    """
     Takes one 3-dimensional array with the constant value of padding.
     Your task is to pad the instances with the given constant value while maintaining the array at the center of the padding.
    :return:
    """
    max_batch = (max([batch.shape[0] for batch in array]), max([batch.shape[1] for batch in array]))
    for batch in array:
        before_x, before_y = (max_batch[0] - batch.shape[0]) / 2, (max_batch[1] - batch.shape[1]) / 2
        after_x, after_y = max_batch[0] - batch.shape[0] - before_x, max_batch[1] - batch.shape[1] - before_y
        batch = numpy.pad(batch, ((before_x, after_x), (before_y, after_y)), "constant", constant_value=acval)
    return array


#######################################
# PyTorch
#######################################

# numpy&torch

def numpy2tensor(array):
    """
    Takes a numpy ndarray and converts it to a PyTorch tensor.
    Function torch.tensor is one of the simple ways to implement it but please do not use it this time.
    :return:
    """
    return torch.from_numpy(array)


def tensor2numpy(array):
    """
    Takes a PyTorch tensor and converts it to a numpy ndarray.
    :return:
    """
    return array.numpy()


# Tensor Sum-products

def Tensor_Sumproducts(tensor1, tensor2):
    """
    you are to implement the function tensor sumproducts that takes two tensors as input.
    returns the sum of the element-wise products of the two tensors.
    :return:
    """
    return tensor1 + tensor2


# Tensor ReLu and ReLu prime

def Tensor_Relu(tensor1):
    """
    Takes one 2-dimensional tensor and apply the relu function on all the values of the tensor.
    :return:
    """
    return torch.nn.functional.relu(tensor1)


def Tensor_Relu_prime(tensor1):
    """
    Takes one 2-dimensional tensor and apply the derivative of relu function on all the values of the tensor.
    :return:
    """
    return torch.nn.functional.relu(tensor1)


if __name__ == "__main__":
    import numpy

    sparseinput = [[[1., 1.], [2.], [1.2]], [[2.3, 4.0], [6., 7, 8., 9, ]]]
    input = torch.randn(3, 3, 3).numpy()
    print(input)
    print(slice_fixed_point(sparseinput, (0, 0, 0), (1, 2, 2)))
    print(slice_last_point(sparseinput, 1))

    print(slice_random_point(input, (2, 1)))
