import argparse
import tensorflow as tf
import numpy as np
import imageio
import elpips


# Ensure TensorFlow uses GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(physical_devices)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
    except RuntimeError as e:
        print(e)

# Command line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, nargs=2, help='two images whose distance needs to be evaluated')
parser.add_argument('--metric', type=str, default='elpips_vgg', help='elpips_vgg (default), lpips_vgg or lpips_squeeze')
parser.add_argument('-n', type=int, default=200, help='number of samples to use for E-LPIPS. Default: 200')
args = parser.parse_args()

if args.metric not in ('elpips_vgg', 'lpips_vgg', 'lpips_squeeze'):
    raise Exception('Unsupported metric')

# Load images.
image1 = imageio.v2.imread(args.image[0])[:,:,0:3].astype(np.float32) / 255.0
image2 = imageio.v2.imread(args.image[1])[:,:,0:3].astype(np.float32) / 255.0

assert image1.shape == image2.shape

# Create the distance metric.
if args.metric == 'elpips_vgg':
    # Use E-LPIPS averages over n samples.
    metric = elpips.Metric(elpips.elpips_vgg(batch_size=1, n=args.n), back_prop=False)
elif args.metric == 'lpips_vgg':
    # Use LPIPS-VGG.
    metric = elpips.Metric(elpips.lpips_vgg(1), back_prop=False)
elif args.metric == 'lpips_squeeze':
    # Use LPIPS-SQUEEZENET.
    metric = elpips.Metric(elpips.lpips_squeeze(1), back_prop=False)
else:
    raise Exception('Unsupported metric')

# Convert images to tensors
tf_image1 = tf.convert_to_tensor(np.expand_dims(image1, axis=0), dtype=tf.float32)
tf_image2 = tf.convert_to_tensor(np.expand_dims(image2, axis=0), dtype=tf.float32)

# Compute the distance using the metric
print("Creating computation graph.")

@tf.function
def compute_distance(img1, img2):
    return metric.forward(img1, img2)

# Run.
print("Running graph.")
distances_in_minibatch = compute_distance(tf_image1, tf_image2)

# Check device placement
print(tf_image1.device)
print(tf_image2.device)
print(distances_in_minibatch.device)

print("Distance ({}): {}".format(args.metric, distances_in_minibatch[0].numpy()))
