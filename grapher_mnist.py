import matplotlib.pyplot as plt
import numpy as np


INITIAL_TRAIN_SET_SIZE = 12000
#epoch:10 and batch_size: 5000
BATCH_SIZE_5K = 5000
asce_b5k_e10 = [0.897, 0.9343, 0.9491, 0.9583, 0.9651, 0.9702, 0.9721, 0.9722, 0.9757]
desc_b5k_e10 = [0.9048, 0.9141, 0.9194, 0.9255, 0.9289, 0.9282, 0.9281, 0.9281, 0.9287]
rand_b5k_e10 = [0.9222, 0.9451, 0.9565, 0.9596, 0.9608, 0.9642, 0.9645, 0.9639, 0.9648]
batch_5k_x_axis = list(np.arange(0,len(asce_b5k_e10))*BATCH_SIZE_5K + INITIAL_TRAIN_SET_SIZE)

#epoch:10 and batch_size: 2000
BATCH_SIZE_2K = 2000
asce_b2k_e10 = [0.9154, 0.9464, 0.9581, 0.9664, 0.9731, 0.9741, 0.9762, 0.9777, 0.979, 0.9776, 0.9794, 0.9803, 0.9802, 0.9795, 0.9797, 0.9808, 0.9809, 0.9796, 0.9806, 0.9802, 0.9802, 0.9805, 0.981, 0.9814]
desc_b2k_e10 = [0.9011, 0.9288, 0.9403, 0.943, 0.9456, 0.9487, 0.9473, 0.9483, 0.9429, 0.9491, 0.9449, 0.9444, 0.9486, 0.9478, 0.9475, 0.947, 0.9476, 0.9475, 0.948, 0.9486, 0.9501, 0.9475, 0.9492, 0.9442]
rand_b2k_e10 = [0.9101, 0.9385, 0.9499, 0.9616, 0.9633, 0.9676, 0.9676, 0.9703, 0.9694, 0.9715, 0.9697, 0.9718, 0.9718, 0.9705, 0.9715, 0.9699, 0.9707, 0.9708, 0.9697, 0.9705, 0.9693, 0.969, 0.9698, 0.9688]
batch_2k_x_axis = list(np.arange(0,len(asce_b2k_e10))*BATCH_SIZE_2K + INITIAL_TRAIN_SET_SIZE)


#epoch:10 and batch_size: 1000
BATCH_SIZE_1K = 1000
asce_b1k_e10 = [0.906, 0.9352, 0.9533, 0.9601, 0.9654, 0.9704, 0.9731, 0.9754, 0.9754, 0.9767, 0.978, 0.9791, 0.9789, 0.9789, 0.9784, 0.98, 0.9804, 0.982, 0.9807, 0.9811, 0.9809, 0.9815, 0.9816, 0.9817, 0.9822, 0.9819, 0.9826, 0.9824, 0.9824, 0.9826, 0.9826, 0.9826, 0.9821, 0.9824, 0.9818, 0.9821, 0.9817, 0.982, 0.9823, 0.9825, 0.9824, 0.9827, 0.9827, 0.9823, 0.9828, 0.9828, 0.9832, 0.9833]
desc_b1k_e10 = [0.9065, 0.9329, 0.9464, 0.9554, 0.9596, 0.9628, 0.9637, 0.9649, 0.9654, 0.967, 0.9666, 0.9653, 0.9672, 0.9677, 0.9664, 0.9648, 0.9683, 0.9669, 0.9687, 0.9659, 0.9681, 0.9697, 0.9681, 0.968, 0.968, 0.97, 0.9687, 0.9686, 0.9676, 0.9673, 0.967, 0.9677, 0.9694, 0.9678, 0.9643, 0.9646, 0.9651, 0.9655, 0.9662, 0.9666, 0.9655, 0.9669, 0.9658, 0.9669, 0.9665, 0.9661, 0.9665, 0.9659]
rand_b1k_e10 = [0.907, 0.9415, 0.9524, 0.9609, 0.9665, 0.9665, 0.971, 0.9735, 0.9735, 0.9728, 0.9736, 0.9739, 0.9755, 0.9755, 0.9756, 0.9756, 0.9745, 0.9757, 0.9766, 0.9748, 0.9716, 0.9749, 0.9729, 0.9738, 0.976, 0.9743, 0.9753, 0.9733, 0.9752, 0.9734, 0.9744, 0.9749, 0.9736, 0.9759, 0.9733, 0.9745, 0.9752, 0.9752, 0.9748, 0.975, 0.9751, 0.9747, 0.9742, 0.9745, 0.9748, 0.9747, 0.9742, 0.9751]
batch_1k_x_axis = list(np.arange(0,len(asce_b1k_e10))*BATCH_SIZE_1K + INITIAL_TRAIN_SET_SIZE)

#first plot
fig, ax = plt.subplots()
line1, = ax.plot(batch_5k_x_axis, asce_b5k_e10 , marker='o', color='b',label='ASC')
line2, = ax.plot(batch_5k_x_axis, desc_b5k_e10 , marker='^', color='g',label='DESC')
line3, = ax.plot(batch_5k_x_axis, rand_b5k_e10 , marker='s', color='r',label='RAND')

plt.xticks(batch_5k_x_axis)
plt.xlabel("Number of training images")
plt.ylabel("Accuracy over 10k images")
ax.legend()
plt.savefig("e10_b5k.png")
plt.show()

#second plot

fig, ax = plt.subplots()
line1, = ax.plot(batch_2k_x_axis, asce_b2k_e10 , marker='o', color='b',label='ASC')
line2, = ax.plot(batch_2k_x_axis, desc_b2k_e10 , marker='^', color='g',label='DESC')
line3, = ax.plot(batch_2k_x_axis, rand_b2k_e10 , marker='s', color='r',label='RAND')

plt.xticks(batch_2k_x_axis)
plt.xlabel("Number of training images")
plt.ylabel("Accuracy over 10k test images")
ax.legend()
plt.savefig("e10_b2k.png")
plt.show()

#third plot
fig, ax = plt.subplots()
line1, = ax.plot(batch_1k_x_axis, asce_b1k_e10 , marker='o', color='b',label='ASC')
line2, = ax.plot(batch_1k_x_axis, desc_b1k_e10 , marker='^', color='g',label='DESC')
line3, = ax.plot(batch_1k_x_axis, rand_b1k_e10 , marker='s', color='r',label='RAND')

plt.xticks(batch_1k_x_axis)
plt.xlabel("Number of training images")
plt.ylabel("Accuracy over 10k test images")
ax.legend()
plt.savefig("e10_b1k.png")
plt.show()
