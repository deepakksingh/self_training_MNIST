import matplotlib.pyplot as plt
import numpy as np


INITIAL_TRAIN_SET_SIZE = 12000
#epoch:10 and batch_size: 5000
BATCH_SIZE_5K = 5000
asce_b5k_e10 = [0.897, 0.9343, 0.9491, 0.9583, 0.9651, 0.9702, 0.9721, 0.9722, 0.9757]
desc_b5k_e10 = [0.9048, 0.9141, 0.9194, 0.9255, 0.9289, 0.9282, 0.9281, 0.9281, 0.9287]
rand_b5k_e10 = [0.9222, 0.9451, 0.9565, 0.9596, 0.9608, 0.9642, 0.9645, 0.9639, 0.9648]
base_b5k_e10 = [0.9122, 0.9504, 0.9702, 0.9774, 0.9789, 0.9847, 0.9857, 0.9873, 0.9879]
batch_5k_x_axis = list(np.arange(0,len(asce_b5k_e10))*BATCH_SIZE_5K + INITIAL_TRAIN_SET_SIZE)

#epoch:10 and batch_size: 2000
BATCH_SIZE_2K = 2000
asce_b2k_e10 = [0.9154, 0.9464, 0.9581, 0.9664, 0.9731, 0.9741, 0.9762, 0.9777, 0.979, 0.9776, 0.9794, 0.9803, 0.9802, 0.9795, 0.9797, 0.9808, 0.9809, 0.9796, 0.9806, 0.9802, 0.9802, 0.9805, 0.981, 0.9814]
desc_b2k_e10 = [0.9011, 0.9288, 0.9403, 0.943, 0.9456, 0.9487, 0.9473, 0.9483, 0.9429, 0.9491, 0.9449, 0.9444, 0.9486, 0.9478, 0.9475, 0.947, 0.9476, 0.9475, 0.948, 0.9486, 0.9501, 0.9475, 0.9492, 0.9442]
rand_b2k_e10 = [0.9101, 0.9385, 0.9499, 0.9616, 0.9633, 0.9676, 0.9676, 0.9703, 0.9694, 0.9715, 0.9697, 0.9718, 0.9718, 0.9705, 0.9715, 0.9699, 0.9707, 0.9708, 0.9697, 0.9705, 0.9693, 0.969, 0.9698, 0.9688]
base_b2k_e10 = [0.9042, 0.9518, 0.9658, 0.9741, 0.9746, 0.9807, 0.983, 0.984, 0.9855, 0.9856, 0.9864, 0.9872, 0.9879, 0.9886, 0.9891, 0.9881, 0.9883, 0.9889, 0.9892, 0.9895, 0.99, 0.9898, 0.9901, 0.9901]
batch_2k_x_axis = list(np.arange(0,len(asce_b2k_e10))*BATCH_SIZE_2K + INITIAL_TRAIN_SET_SIZE)


#epoch:10 and batch_size: 1000
BATCH_SIZE_1K = 1000
asce_b1k_e10 = [0.906, 0.9352, 0.9533, 0.9601, 0.9654, 0.9704, 0.9731, 0.9754, 0.9754, 0.9767, 0.978, 0.9791, 0.9789, 0.9789, 0.9784, 0.98, 0.9804, 0.982, 0.9807, 0.9811, 0.9809, 0.9815, 0.9816, 0.9817, 0.9822, 0.9819, 0.9826, 0.9824, 0.9824, 0.9826, 0.9826, 0.9826, 0.9821, 0.9824, 0.9818, 0.9821, 0.9817, 0.982, 0.9823, 0.9825, 0.9824, 0.9827, 0.9827, 0.9823, 0.9828, 0.9828, 0.9832, 0.9833]
desc_b1k_e10 = [0.9065, 0.9329, 0.9464, 0.9554, 0.9596, 0.9628, 0.9637, 0.9649, 0.9654, 0.967, 0.9666, 0.9653, 0.9672, 0.9677, 0.9664, 0.9648, 0.9683, 0.9669, 0.9687, 0.9659, 0.9681, 0.9697, 0.9681, 0.968, 0.968, 0.97, 0.9687, 0.9686, 0.9676, 0.9673, 0.967, 0.9677, 0.9694, 0.9678, 0.9643, 0.9646, 0.9651, 0.9655, 0.9662, 0.9666, 0.9655, 0.9669, 0.9658, 0.9669, 0.9665, 0.9661, 0.9665, 0.9659]
rand_b1k_e10 = [0.907, 0.9415, 0.9524, 0.9609, 0.9665, 0.9665, 0.971, 0.9735, 0.9735, 0.9728, 0.9736, 0.9739, 0.9755, 0.9755, 0.9756, 0.9756, 0.9745, 0.9757, 0.9766, 0.9748, 0.9716, 0.9749, 0.9729, 0.9738, 0.976, 0.9743, 0.9753, 0.9733, 0.9752, 0.9734, 0.9744, 0.9749, 0.9736, 0.9759, 0.9733, 0.9745, 0.9752, 0.9752, 0.9748, 0.975, 0.9751, 0.9747, 0.9742, 0.9745, 0.9748, 0.9747, 0.9742, 0.9751]
base_b1k_e10 = [0.9081, 0.9406, 0.9568, 0.9662, 0.9728, 0.9753, 0.9787, 0.9819, 0.9824, 0.9822, 0.9861, 0.9857, 0.9872, 0.9857, 0.9861, 0.9882, 0.9873, 0.9875, 0.9869, 0.9885, 0.9875, 0.9888, 0.9892, 0.9885, 0.9885, 0.9895, 0.99, 0.9901, 0.9902, 0.9897, 0.989, 0.9899, 0.9898, 0.9896, 0.9895, 0.9898, 0.99, 0.9895, 0.9898, 0.99, 0.9899, 0.9901, 0.9902, 0.9903, 0.9903, 0.9901, 0.9904, 0.9904]
batch_1k_x_axis = list(np.arange(0,len(asce_b1k_e10))*BATCH_SIZE_1K + INITIAL_TRAIN_SET_SIZE)

#epoch:1 and batch_size: 5000
BATCH_SIZE_5K = 5000
asce_b5k_e1 = [0.5809, 0.5815, 0.8145, 0.8514, 0.8758, 0.8895, 0.8964, 0.9055, 0.9147]
desc_b5k_e1 = [0.516, 0.4425, 0.5881, 0.6135, 0.6213, 0.6541, 0.6456, 0.6447, 0.6361]
rand_b5k_e1 = [0.4865, 0.2873, 0.4489, 0.5071, 0.5209, 0.4974, 0.5531, 0.619, 0.5393]
base_b5k_e1 = [0.3859, 0.7066, 0.8231, 0.8937, 0.9138, 0.931, 0.9398, 0.9489, 0.9548]
batch_5k_x_axis = list(np.arange(0,len(asce_b5k_e1))*BATCH_SIZE_5K + INITIAL_TRAIN_SET_SIZE)

#epoch:1 and batch_size: 2000
BATCH_SIZE_2K = 2000
asce_b2k_e1 = [0.5613, 0.5386, 0.7888, 0.8528, 0.8758, 0.8891, 0.8979, 0.9026, 0.9082, 0.9125, 0.9166, 0.9218, 0.9228, 0.9265, 0.9277, 0.9277, 0.931, 0.9363, 0.9399, 0.9401, 0.9442, 0.9459, 0.9491, 0.9522]
desc_b2k_e1 = [0.4861, 0.4941, 0.6776, 0.7226, 0.8061, 0.8053, 0.7934, 0.8094, 0.8053, 0.8272, 0.8272, 0.8133, 0.8186, 0.8357, 0.83, 0.8413, 0.8326, 0.8333, 0.8302, 0.8363, 0.8304, 0.8328, 0.8307, 0.8418]
rand_b2k_e1 = [0.4014, 0.4099, 0.5856, 0.7049, 0.7877, 0.7926, 0.8175, 0.808, 0.8145, 0.843, 0.8385, 0.8399, 0.8299, 0.8399, 0.8569, 0.859, 0.8597, 0.857, 0.8638, 0.862, 0.8594, 0.8596, 0.8755, 0.8648]
base_b2k_e1 = [0.484, 0.754, 0.8362, 0.8721, 0.8943, 0.9032, 0.9151, 0.9245, 0.9308, 0.9362, 0.9435, 0.9484, 0.9532, 0.956, 0.9594, 0.9608, 0.9663, 0.9672, 0.9701, 0.9698, 0.9737, 0.9752, 0.976, 0.9769]
batch_2k_x_axis = list(np.arange(0,len(asce_b2k_e1))*BATCH_SIZE_2K + INITIAL_TRAIN_SET_SIZE)


#epoch:10 and batch_size: 1000
BATCH_SIZE_1K = 1000
asce_b1k_e1 = [0.4347, 0.3708, 0.7252, 0.8083, 0.8427, 0.8687, 0.8835, 0.8895, 0.8956, 0.904, 0.9064, 0.9135, 0.9143, 0.9208, 0.9249, 0.9291, 0.9251, 0.9342, 0.9335, 0.9373, 0.9394, 0.9399, 0.944, 0.9448, 0.9492, 0.9513, 0.9497, 0.952, 0.9553, 0.9546, 0.9543, 0.9506, 0.9561, 0.9599, 0.9603, 0.9607, 0.9595, 0.9604, 0.9625, 0.9627, 0.9628, 0.9641, 0.9662, 0.9659, 0.9648, 0.9651, 0.968, 0.9681]
desc_b1k_e1 = [0.3696, 0.4357, 0.6711, 0.7579, 0.7861, 0.8156, 0.7984, 0.8283, 0.808, 0.826, 0.8154, 0.8014, 0.8085, 0.8031, 0.838, 0.807, 0.8245, 0.8168, 0.8152, 0.8259, 0.8229, 0.8412, 0.8107, 0.8285, 0.8307, 0.8273, 0.8228, 0.8181, 0.8234, 0.8205, 0.8239, 0.8234, 0.8307, 0.8294, 0.8348, 0.8309, 0.8378, 0.8322, 0.8391, 0.8285, 0.8283, 0.825, 0.8229, 0.8279, 0.8242, 0.8201, 0.839, 0.8285]
rand_b1k_e1 = [0.5106, 0.6224, 0.7061, 0.8128, 0.8492, 0.8702, 0.8821, 0.8893, 0.8813, 0.8961, 0.9005, 0.8949, 0.9084, 0.9152, 0.9031, 0.9139, 0.9115, 0.919, 0.9172, 0.9229, 0.9245, 0.9264, 0.92, 0.9269, 0.9271, 0.9279, 0.9251, 0.9258, 0.9336, 0.9321, 0.9328, 0.9348, 0.9363, 0.9305, 0.9381, 0.9237, 0.9301, 0.9326, 0.9408, 0.9368, 0.9375, 0.932, 0.9341, 0.9388, 0.9389, 0.937, 0.9381, 0.9372]
base_b1k_e1 = [0.4163, 0.7134, 0.8049, 0.866, 0.883, 0.9017, 0.9133, 0.9179, 0.9227, 0.9308, 0.9336, 0.9405, 0.9434, 0.9467, 0.9508, 0.954, 0.9562, 0.9595, 0.9614, 0.9619, 0.9643, 0.9656, 0.9678, 0.9698, 0.9715, 0.9727, 0.9722, 0.9734, 0.9755, 0.9754, 0.977, 0.9775, 0.9789, 0.9791, 0.9788, 0.9788, 0.9806, 0.9818, 0.9826, 0.9827, 0.983, 0.9826, 0.9837, 0.9833, 0.9835, 0.9826, 0.9836, 0.9851]
batch_1k_x_axis = list(np.arange(0,len(asce_b1k_e1))*BATCH_SIZE_1K + INITIAL_TRAIN_SET_SIZE)

# batch_5k_x_labels = list(np.arange(12000,60000,10000))
# print(batch_5k_x_labels)

#first plot
fix, (ax1,ax2) = plt.subplots(1, 2)
line1, = ax1.plot(batch_5k_x_axis, asce_b5k_e10 , marker='o', color='b',label='ASC')
line2, = ax1.plot(batch_5k_x_axis, desc_b5k_e10 , marker='^', color='g',label='DESC')
line3, = ax1.plot(batch_5k_x_axis, rand_b5k_e10 , marker='s', color='r',label='RAND')
line4, = ax1.plot(batch_5k_x_axis, base_b5k_e10 , marker='o', color='k',label='BASE')

ax1.set_title("Model trained on 10 epoch(s) with batch size of 5k")
ax1.set_xticklabels(batch_5k_x_axis, rotation="vertical")
ax1.set_xlabel("Number of training images")
ax1.set_ylabel("Accuracy on 10k test images")
ax1.legend()
ax1.grid()


line1, = ax2.plot(batch_5k_x_axis, asce_b5k_e1 , marker='o', color='b',label='ASC')
line2, = ax2.plot(batch_5k_x_axis, desc_b5k_e1 , marker='^', color='g',label='DESC')
line3, = ax2.plot(batch_5k_x_axis, rand_b5k_e1 , marker='s', color='r',label='RAND')
line4, = ax2.plot(batch_5k_x_axis, base_b5k_e1 , marker='o', color='k',label='BASE')

ax2.set_title("Model trained on 1 epoch(s) with batch size of 5k")
ax2.set_xticklabels(batch_5k_x_axis, rotation="vertical")
ax2.set_xlabel("Number of training images")
ax2.set_ylabel("Accuracy on 10k test images")
ax2.legend()
ax2.grid()

plt.subplots_adjust(
    top=0.948,
    bottom=0.131,
    left=0.055,
    right=0.986,
    hspace=0.2,
    wspace=0.116
)
# plt.savefig("b5k_combined.png", dpi=500)

plt.show()

#second plot
fix, (ax1,ax2) = plt.subplots(1, 2)
line1, = ax1.plot(batch_2k_x_axis, asce_b2k_e10 , marker='o', color='b',label='ASC')
line2, = ax1.plot(batch_2k_x_axis, desc_b2k_e10 , marker='^', color='g',label='DESC')
line3, = ax1.plot(batch_2k_x_axis, rand_b2k_e10 , marker='s', color='r',label='RAND')
line4, = ax1.plot(batch_2k_x_axis, base_b2k_e10 , marker='o', color='k',label='BASE')

ax1.set_title("Model trained on 10 epoch(s) with batch size of 2k")
ax1.set_xticklabels(batch_2k_x_axis, rotation="vertical")
ax1.set_xlabel("Number of training images")
ax1.set_ylabel("Accuracy on 10k test images")
ax1.legend()
ax1.grid()


line1, = ax2.plot(batch_2k_x_axis, asce_b2k_e1 , marker='o', color='b',label='ASC')
line2, = ax2.plot(batch_2k_x_axis, desc_b2k_e1 , marker='^', color='g',label='DESC')
line3, = ax2.plot(batch_2k_x_axis, rand_b2k_e1 , marker='s', color='r',label='RAND')
line4, = ax2.plot(batch_2k_x_axis, base_b2k_e1 , marker='o', color='k',label='BASE')

ax2.set_title("Model trained on 1 epoch(s) with batch size of 2k")
ax2.set_xticklabels(batch_2k_x_axis, rotation="vertical")
ax2.set_xlabel("Number of training images")
ax2.set_ylabel("Accuracy on 10k test images")
ax2.legend()
ax2.grid()

plt.subplots_adjust(
    top=0.948,
    bottom=0.131,
    left=0.055,
    right=0.986,
    hspace=0.2,
    wspace=0.116
)
# plt.savefig("b2k_combined.png", dpi=500)
plt.show()

#third plot
fix, (ax1,ax2) = plt.subplots(1, 2)
line1, = ax1.plot(batch_1k_x_axis, asce_b1k_e10 , marker='o', color='b',label='ASC')
line2, = ax1.plot(batch_1k_x_axis, desc_b1k_e10 , marker='^', color='g',label='DESC')
line3, = ax1.plot(batch_1k_x_axis, rand_b1k_e10 , marker='s', color='r',label='RAND')
line4, = ax1.plot(batch_1k_x_axis, base_b1k_e10 , marker='o', color='k',label='BASE')

ax1.set_title("Model trained on 10 epoch(s) with batch size of 1k")
ax1.set_xticklabels(batch_1k_x_axis, rotation="vertical")
ax1.set_xlabel("Number of training images")
ax1.set_ylabel("Accuracy on 10k test images")
ax1.legend()
ax1.grid()


line1, = ax2.plot(batch_1k_x_axis, asce_b1k_e1 , marker='o', color='b',label='ASC')
line2, = ax2.plot(batch_1k_x_axis, desc_b1k_e1 , marker='^', color='g',label='DESC')
line3, = ax2.plot(batch_1k_x_axis, rand_b1k_e1 , marker='s', color='r',label='RAND')
line4, = ax2.plot(batch_1k_x_axis, base_b1k_e1 , marker='o', color='k',label='BASE')

ax2.set_title("Model trained on 1 epoch(s) with batch size of 1k")
ax2.set_xticklabels(batch_1k_x_axis, rotation="vertical")
ax2.set_xlabel("Number of training images")
ax2.set_ylabel("Accuracy on 10k test images")
ax2.legend()
ax2.grid()

plt.subplots_adjust(
    top=0.948,
    bottom=0.131,
    left=0.055,
    right=0.986,
    hspace=0.2,
    wspace=0.116   
)
# plt.savefig("b1k_combined.png", dpi=500)
plt.show()

