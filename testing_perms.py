import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from madnis.transforms.permutation import SoftPermuteLearn as SPL

tf.keras.backend.set_floatx("float64")

# test initialization and orthogonality
test_dim = 20
tmp = SPL((test_dim,))

orth_test = tf.linalg.matmul(tmp._translate_to_matrix(),
                             tmp._translate_to_matrix(), transpose_a=True)

print("Matrix of dimension {} times its transpose is the identity:".format(test_dim))
print(np.allclose(np.eye(test_dim), orth_test.numpy()))
#print("Matrix of dimension {} times its defined inverse is the identity:".format(test_dim))
#print(np.allclose(np.eye(test_dim), orth_test2.numpy()))

# now check if gradients work in 2.
# initialize random rotation and try to let it learn the identity


dim = 6
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-2)
rotation = SPL((dim,))
#vector = tf.constant([1., 1., 1., 1., 1.], dtype=tf.float64)
all_angles = []

print("rotation: ", rotation._translate_to_matrix(), "\n")
#print("vector: ", vector, "\n")
#print("rotatet vector: ", rotation(vector), "\n")

print("weights", rotation.trainable_weights, "\n")

num_epochs = 500
for i in range(num_epochs):
    #print("Epoch {}, angle = {}".format(i+1, rotation.perm_ang))
    print("Epoch {}/{}".format(i+1, num_epochs))
    all_angles.append(rotation.perm_ang)

    with tf.GradientTape() as tape:
        vector = tf.constant(2.*np.random.randn(5000, dim), dtype=tf.float64)
        rotated_vector, _ = rotation(vector)
        difference = (vector - rotated_vector)**2

        loss = tf.math.reduce_sum(difference)

        print("loss: ", loss.numpy(), "\n")

    grads = tape.gradient(loss, rotation.trainable_weights)
    #print("grads: ", grads)
    optimizer.apply_gradients(zip(grads, rotation.trainable_weights))

print("Final angle(s) = {}".format(rotation.perm_ang))
all_angles.append(rotation.perm_ang)
all_angles = np.array(all_angles).T
print("rotation: ", rotation._translate_to_matrix(), "\n")
x_scan = np.arange(501)
for idx, angles in enumerate(all_angles):
    plt.plot(x_scan, angles, label=r"$\alpha-$"+str(idx))
plt.xlabel("epoch")
plt.ylabel("Euler angles")
plt.plot(x_scan, np.zeros_like(x_scan), label='target', color='k', ls='dashed')
plt.legend()
plt.show()
