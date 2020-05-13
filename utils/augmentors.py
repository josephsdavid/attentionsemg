import numpy as np
import random


def roll_labels(x, y):
    labs_rolled = []
    for i in range(len(y)):
        l = y[i]
        n = x[i].shape[0]
        labs_rolled.append(np.repeat(l,n))
    return np.hstack(labs_rolled)

def window_roll(a, stepsize=5, width=52):
    n = a.shape[0]
    emg =  np.dstack( [a[i:1+n+i-width:stepsize] for i in range(0,width)] )
    return emg


"""
augmentors should take in a list of arrays and ouptut a list of bigger arrays,
arguments should be (X, augmentors should take in a list of arrays and ouptut a list of bigger arrays,
arguments should be (X, y)

"""


def add_noise_snr(signal, snr = 25):
    # convert signal to db
    sgn_db = np.log10((signal ** 2).mean(axis = 0))  * 10
    # noise in db
    noise_avg_db = sgn_db - snr
    # convert noise_db
    noise_variance = 10 ** (noise_avg_db /10)
    # make some white noise using this as std
    noise = np.random.normal(0, np.sqrt(noise_variance), signal.shape)
    return(signal + noise)

rlist = sum([[(x/2)%30]*((x//2)%30) for x in range(120)], [])
def add_noise_random(signal):
    num = random.choice(rlist)
    return add_noise_snr(signal, num)

def add_noise(x, y, snr=25):
    x2 = []
    for i in range(len(x)):
        x2.append(add_noise_snr(x[i]))
    x = x + x2
    y = y*2
    return x, y


def shift_electrodes(examples, labels):
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active canals for those classes
    for classe in range(3, 7):
        X_example = []
        Y_example = []
        for k in range(len(examples)):
            X_example.append(examples[k])
            Y_example.append(labels[k])

        cwt_add = []
        for j in range(len(X_example)):
            if Y_example[j] == classe:
                if cwt_add == []:
                    cwt_add = np.array(X_example[j][0])
                else:
                    cwt_add += np.array(X_example[j][0])
        class_mean.append(np.argmax(np.sum(np.array(cwt_add), axis=0)))

    # We check how many we have to shift for each channels to get back to the normal activation
    new_cwt_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10)
    new_cwt_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10)

    shifts_array = []
    for valueA, valueB in zip(new_cwt_emplacement_left, new_cwt_emplacement_right):
        if valueA < valueB:
            # We want to shift toward the left (the start of the array)
            orientation = -1
            shifts_array.append(orientation*valueA)
        else:
            # We want to shift toward the right (the end of the array)
            orientation = 1
            shifts_array.append(orientation*valueB)

    # We get the mean amount of shift and round it up to get a discrete number representing how much we have to shift
    # if we consider all the canals
    # Do the shifting only if the absolute mean is greater or equal to 0.5
    final_shifting = np.mean(np.array(shifts_array))
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    # Build the dataset of the candiate with the circular shift taken into account.
    X_example = []
    Y_example = []
    for k in range(len(examples)):
        sub_ensemble_example = []
        for example in examples[k]:
            sub_ensemble_example.append(np.roll(np.array(example), final_shifting))
        X_example.append(sub_ensemble_example)
        Y_example.append(labels[k])
    return X_example, Y_example


def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def shift_chance(x,value):
    out = np.roll(x,random.randrange(-15, 16), axis=-1) if random.random() < value else x
    return out
def shift_maybe(arr, chance_value=0.5):
    return np.array([shift_chance(arr[i,:], chance_value) for i in range(arr.shape[0])])

def shift_random(arr):
    return shift_chance(arr, 0.9)
