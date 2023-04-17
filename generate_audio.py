import argparse
import os
import tensorflow as tf
import numpy as np
import wavegan

def generate_audio(model_path, output_file, duration_seconds):
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_path, 'train', 'infer', 'infer.meta'))
        checkpoint = tf.train.latest_checkpoint(os.path.join(model_path, 'train'))
        
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
        else:
            raise ValueError("No valid checkpoint found.")
        
        sample_rate = 16000  # Set the sample rate directly
        slice_duration = 4  # WaveGAN generates 4-second slices
        num_slices = int(duration_seconds / slice_duration)
        
        G_z = sess.graph.get_tensor_by_name('G_z:0')[:, :, 0]
        
        audio_slices = []
        for _ in range(num_slices):
            G_z_spec = sess.run(G_z, feed_dict={'z:0': np.random.normal(size=(1, 100))})
            audio_slices.append(G_z_spec)
        
        audio_concat = np.concatenate(audio_slices, axis=1)[:sample_rate * duration_seconds]
        audio_concat = np.expand_dims(audio_concat, axis=-1)  # Reshape to (num_samples, 1)
        audio = sess.run(tf.audio.encode_wav(audio_concat, sample_rate))
        
        with open(output_file, 'wb') as f:
            f.write(audio)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='.', help='Path to the model directory')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--duration_seconds', type=int, default=60, help='Duration of the generated audio in seconds')
    args = parser.parse_args()
    
    generate_audio(args.model_path, args.output_file, args.duration_seconds)
