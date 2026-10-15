"""Execute official Ear-for-Face-SR weights on CPU with TF1 compatibility.

This is a single-image audio-restoration compatibility/intervention probe, NOT
a temporal restoration or active-speaker experiment. No speaking labels exist
for these author-supplied still-image examples. No research accuracy is emitted.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'external' / 'ear_for_face_super_resolution'


class NetworkMetadata:
    def __setstate__(self, state):
        self.state = state


class MetadataUnpickler(pickle.Unpickler):
    """Read configuration without executing the model's pickled Python source."""
    def find_class(self, module, name):
        if name == 'Network' and module in ('dnnlib.tflib.network', 'tfutil'):
            return NetworkMetadata
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
        }
        if (module, name) in allowed:
            return allowed[module, name]
        raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=int, default=1)
    parser.add_argument('--unitalk', action='store_true', help='Run the preregistered labeled framewise intervention')
    args = parser.parse_args()
    dest = ROOT / 'data' / 'ear_cpu'
    dest.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    print('Importing TensorFlow CPU runtime', flush=True)
    import tensorflow as tensorflow
    import tf_slim
    tf = tensorflow.compat.v1
    tf.disable_v2_behavior()
    tf.contrib = types.SimpleNamespace(layers=tf_slim)
    # Upstream imports receive TF1-compatible APIs; upstream files are unchanged.
    sys.modules['tensorflow'] = tf
    sys.path.insert(0, str(UPSTREAM))
    import networks
    import networks_stylegan as style

    with (UPSTREAM / 'checkpoint' / 'network-snapshot-030929.pkl').open('rb') as stream:
        metadata = MetadataUnpickler(stream).load()
    config = dict(metadata[2].state['static_kwargs'])
    name = metadata[2].state['name']
    print('Original generator configuration:', name, config, flush=True)
    del metadata
    gc.collect()

    tf.set_random_seed(0)
    im = tf.placeholder(tf.float32, [1, 8, 8, 3], name='input_low_resolution')
    au = tf.placeholder(tf.float32, [1, 257, 257, 1], name='input_audio_spectrogram')
    with tf.device('/CPU:0'):
        with tf.variable_scope('LOW_RES_ENCODER'):
            visual = networks.LowResEncoder(im, resolution=8, batch_size=1, num_scales=3, n_filters=128, output_feature_size=512)
        with tf.variable_scope('AUDIO_SPECTROGRAM_ENCODER'):
            audio = networks.SpectrogramEncoder(au, resolution=257, batch_size=1, num_scales=6, n_filters=64, output_feature_size=512)
        with tf.variable_scope('AUDIO_VISUAL_FUSER'):
            fused = networks.FC3ResidualFuser(tf.reshape(visual, [1, 6144]), tf.reshape(audio, [1, 6144]), batch_size=1)
        latent = tf.placeholder(tf.float32, [1, 12, 512], name='restore_latent')
        # Official test.py unpickles G/D/Gs and then creates another Gs.
        # Its active component scopes are therefore G_mapping_2/G_synthesis_2.
        with tf.variable_scope('G_mapping_2'):
            mapping_config = dict(config, dlatent_broadcast=None)
            mapped = style.G_mapping(tf.reshape(latent, [12, 512]), tf.zeros([12, 0]), **mapping_config)
        with tf.variable_scope('G_synthesis_2', reuse=tf.AUTO_REUSE):
            synthesis_config = dict(config, randomize_noise=False, use_noise=False, is_training=False)
            rendered = style.G_synthesis(tf.reshape(mapped, [1, 12, 512]), **synthesis_config)
            rendered = tf.clip_by_value(tf.transpose(rendered, [0, 2, 3, 1]) / 2 + .5, 0, 1)
        wav_path = tf.placeholder(tf.string, [])
        wav = tf.audio.decode_wav(tf.io.read_file(wav_path), desired_channels=1).audio
        wav.set_shape([64000, 1])
        spec = tensorflow.raw_ops.AudioSpectrogram(input=wav, window_size=512, stride=248, magnitude_squared=True)
        spec = tf.expand_dims(tf.squeeze(tf.math.log(spec + 1e-6), axis=0), 2)
        low, high = tf.reduce_min(spec), tf.reduce_max(spec)
        prepared_audio = tf.expand_dims(2 * (spec + tf.abs(low)) / (high - low) - 1, 0)

    checkpoint = str(UPSTREAM / 'checkpoint' / 'model-0')
    available = dict(tf.train.list_variables(checkpoint))
    variables = {v.name.split(':')[0]: v for v in tf.global_variables()}
    for key, var in variables.items():
        if key not in available or available[key] != var.shape.as_list():
            raise ValueError(f'Checkpoint mismatch: {key}, graph={var.shape}, checkpoint={available.get(key)}')
    # Restore one variable at a time rather than materializing the complete
    # checkpoint's tensors concurrently on this memory-constrained laptop.
    restore_ops = [tf.assign(var, tensorflow.raw_ops.RestoreV2(prefix=checkpoint,
        tensor_names=[key], shape_and_slices=[''], dtypes=[var.dtype.base_dtype])[0])
        for key, var in variables.items()]
    records = []
    options = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
    print('Restoring', len(variables), 'exactly shape-matched variables', flush=True)
    with tf.Session(config=options) as session:
        for op in restore_ops:
            session.run(op)
        print('Checkpoint loaded on CPU', flush=True)
        if args.unitalk:
            from run_unitalk_cpu import run
            run(session, im, au, visual, audio, fused, latent, rendered, wav_path, prepared_audio)
            return
        images = sorted((UPSTREAM / 'test_data' / 'images').glob('*.png'))
        if not 1 <= args.examples <= len(images):
            raise ValueError('Invalid example count')
        for idx, image_path in enumerate(images[:args.examples]):
            original = np.asarray(Image.open(image_path).convert('RGB'))
            # scipy.misc.imresize's legacy RGB float path uses bytescale.
            source = original.astype(np.float64) / 255
            span = source.max() - source.min()
            scaled = ((source - source.min()) * (255 / span if span else 1) + .5).clip(0, 255).astype(np.uint8)
            small = np.asarray(Image.fromarray(scaled).resize((8, 8), Image.Resampling.BICUBIC))
            input_image = small.astype(np.float32)[None] / 127.5 - 1
            matching = UPSTREAM / 'test_data' / 'audio' / (image_path.stem + '.wav')
            donor = UPSTREAM / 'test_data' / 'audio' / (images[(idx + 1) % len(images)].stem + '.wav')
            timings, outputs = {}, {}
            for condition, path in [('video_only', None), ('matching_audio', matching), ('swapped_audio', donor), ('matching_repeat', matching)]:
                started = time.perf_counter()
                if path is None:
                    z = session.run(visual, {im: input_image})
                else:
                    a = session.run(prepared_audio, {wav_path: str(path)})
                    if not np.isfinite(a).all():
                        raise ValueError('Non-finite audio preprocessing')
                    z = session.run(fused, {im: input_image, au: a})
                output = session.run(rendered, {latent: z})[0]
                if not np.isfinite(output).all():
                    raise ValueError('Non-finite output')
                outputs[condition] = (output * 255).round().astype(np.uint8)
                timings[condition] = time.perf_counter() - started
                Image.fromarray(outputs[condition]).save(dest / f'{image_path.stem}_{condition}.png')
                print(image_path.stem, condition, round(timings[condition], 2), 'seconds', flush=True)
            np.savez_compressed(dest / (image_path.stem + '.npz'), reference=original, degraded=small, **outputs)
            row = dict(example=image_path.stem, donor=donor.stem, wall_seconds=timings,
                matching_repeat_equal=bool(np.array_equal(outputs['matching_audio'], outputs['matching_repeat'])),
                audio_swap_rgb_mae=float(np.abs(outputs['matching_audio'].astype(float) - outputs['swapped_audio']).mean()),
                input_image_sha256=file_hash(image_path), matching_audio_sha256=file_hash(matching), donor_audio_sha256=file_hash(donor))
            records.append(row)
    report = dict(status='CPU_FORWARD_PASS_COMPLETED', model='Learning to Have an Ear for Face Super-Resolution',
        upstream_commit='c7f342b1b2984ed674b7407a95305f9cd6f89613', tensorflow=tensorflow.__version__,
        generator_configuration=config, restored_variables=len(variables), device='CPU', examples=records,
        weights_sha256=file_hash(UPSTREAM / 'checkpoint' / 'model-0.data-00000-of-00001'),
        wall_seconds=time.perf_counter() - began,
        limitations=['Single-image model; not a temporal restoration experiment.',
            'TF1 compatibility adaptation; numerical equivalence to the original TF1 GPU runtime has not been established.',
            'No verified speaker labels or downstream error measurements; RQ1-RQ3 are not validated by this probe.'])
    (ROOT / 'results' / 'ear_cpu_probe.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
