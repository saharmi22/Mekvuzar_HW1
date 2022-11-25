import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.array input data

     Returns
     -------
     np.array
         sigmoid of the input x

     """
    L = 1
    k = 1
    x0 = 0
    return np.array([L / (1 + np.exp(-k * (x_elem - x0))) for x_elem in x])


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.array input data

         Returns
         -------
         np.array
             derivative of sigmoid of the input x

    """
    return np.array([val - val*val for val in sigmoid(x)])
    '''
    prime_sig = sigmoid(x)
    for idx, item in enumerate(prime_sig):
        prime_sig[idx] = item*(1-item)
    return prime_sig'''


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of xavier initialized np arrays weight matrices

    """
    return generate_weights(sizes, 'random')


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays weight matrices

    """
    return generate_weights(sizes, 'zeros')

    '''list = []
    for idx, size in enumerate(sizes):
        if idx < len(sizes):
            list.append(np.zeroes((size, sizes[idx+1])))
    return list'''


def zeros_biases(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays bias matrices

    """
    return np.array([]) + [np.zeros(size) for size in sizes[1:]]


#  TODO: fix
def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

    """
    
    
    batch_list = []
    data_labels = np.column_stack((data, labels))#-> [[data1, label1],[data2, label2]....]
    np.random.shuffle(data_labels) #-> shuffled

    data_shuffled = data_labels[:, :-1] #-> all the rows all columns except last one: [data1, data2,...]
    labels_shuffled = data_labels[:, -1] #-> all the rows last column: [label1, label2...]

    for i in range(0, data_shuffled.shape[0], batch_size):
        X_mini = data_shuffled[i:i + batch_size]
        y_mini = labels_shuffled[i:i + batch_size]

        #[(batch1 = [data1,...,data_batch_size],[label1,...,label_batch_size]),...]
        batch_list.append((X_mini, y_mini)) #batch_list contains tuples of data and labels lists. each tuple is a batch

    return batch_list
    
    
    """batch_list = []
    cnt=0
    batch_num = -1
    for data, label in zip(data, labels):
        if cnt == 0:
            batch_num+=1
            batch_list.append(([], []))
        #remove
        batch = list(batch_list.pop(batch_num))
        batch[0].append(data)
        batch[1].append(label)
        tup = tuple(batch)
        batch_list.insert(batch_num, tup)
        cnt+=1
        cnt = divmod(cnt, 16)
    return batch_list"""
    #raise NotImplementedError("To be implemented")


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.array of numbers
         list2 : np.array of numbers

         Returns
         -------
         list
             list of sum of each two elements by index
    """
    assert list1.size == list2.size
    return [list1[i] + list2[i] for i in range(list1.size)]
    # return list1 + list2


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))


#  ******* helper functions: *******


def generate_weights(sizes, distribution):
    f = np.zeros if distribution == 'zeros' else np.random.random if distribution == 'random' else None  # TODO: good?
    '''f = None
    if distribution == 'zeros':
        f = np.zeros
    elif distribution == 'random':
        f = np.random.random'''
    # more? uniform? normal?

    assert f is not None

    return [np.array([])] + [f((sizes[i], sizes[i+1])) for i in range(len(sizes) - 1)]

