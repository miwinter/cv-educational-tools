"""Export Keras 2 VGG16 to TFJS Layers format, unquantized little-endian float32."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KERAS_HOME'] = os.path.abspath('.cache/keras')
import json, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
out = Path('public/model'); out.mkdir(parents=True, exist_ok=True)
model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
specs=[]; chunks=[]
for w in model.weights:
    a=w.numpy().astype('<f4'); specs.append(dict(name=w.name.removesuffix(':0'), shape=list(a.shape), dtype='float32')); chunks.append(a.tobytes())
raw=b''.join(chunks); paths=[]
for i,start in enumerate(range(0,len(raw),4*1024*1024)):
    name=f'weights-{i+1:02}.bin'; (out/name).write_bytes(raw[start:start+4*1024*1024]); paths.append(name)
(out/'model.json').write_text(json.dumps(dict(format='layers-model', generatedBy=f'keras {tf.keras.__version__ if hasattr(tf.keras,"__version__") else tf.__version__}', convertedBy='scripts/prepare_model.py (float32 manifest)', modelTopology=json.loads(model.to_json()), weightsManifest=[dict(paths=paths,weights=specs)])))
# Full intermediate references: deterministic RGB pixels and official preprocessing.
rgb=(np.arange(224*224*3,dtype=np.uint32)*37%256).astype(np.uint8).reshape(1,224,224,3)
x=tf.keras.applications.vgg16.preprocess_input(rgb.astype(np.float32))
ref=Path('tests/reference'); ref.mkdir(exist_ok=True)
(ref/'input.bin').write_bytes(rgb.tobytes()); (ref/'preprocessed.bin').write_bytes(x.astype('<f4').tobytes())
layers=model.layers[1:]; multi=tf.keras.Model(model.input,[l.output for l in layers])
manifest=[]
for layer,y in zip(layers,multi(x,training=False)):
    a=y.numpy().astype('<f4'); (ref/f'{layer.name}.bin').write_bytes(a.tobytes()); manifest.append(dict(name=layer.name,shape=list(a.shape)))
(ref/'manifest.json').write_text(json.dumps(manifest))
(out/'provenance.json').write_text(json.dumps(dict(source='Keras VGG16 ImageNet include_top=False',tensorflow=tf.__version__,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),indent=2))
print('Exported',len(raw),'bytes; 18 full reference outputs')
