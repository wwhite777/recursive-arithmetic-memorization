"""Fresh-batch commitments. Hash ordered indices as unsigned 64-bit LE words."""
import hashlib
import json
from pathlib import Path
import struct

from .common import require

SCHEMA = 'fresh-record-v1'
ALGORITHM = 'torch.randperm-cpu-int64'
ENCODING = 'ordered-uint64-little-endian'


def indices_sha256(indices):
    values = indices.tolist()
    return hashlib.sha256(struct.pack('<'+'Q'*len(values), *values)).hexdigest()


def record_metadata(seed, length, batch_size, total_updates, identities, torch_version):
    return dict(schema=SCHEMA, seed=seed, length=length, batch_size=batch_size,
                total_updates=total_updates, identities=identities, torch_version=str(torch_version),
                algorithm=ALGORITHM, hash_encoding=ENCODING)


def epoch_record(epoch, update, order, generator_state):
    return {'epoch': epoch, 'start_update': update,
            'permutation_sha256': indices_sha256(order),
            'generator_state_sha256': hashlib.sha256(bytes(generator_state.tolist())).hexdigest()}


def batch_record(update, epoch, indices):
    return {'trajectory': update, 'epoch': epoch, 'indices_sha256': indices_sha256(indices)}


def verify_record(root, updates=None):
    """Regenerate from the seed; require every epoch and batch, with no trailing rows.

    Defaults to the complete configured run. An explicit update count verifies a
    partial record (e.g. a calibration), never silently accepts a truncated run.
    No training data, model, checkpoint or CUDA access is needed.
    """
    import torch
    root = Path(root)
    meta = json.loads((root/'FRESH_RECORD.json').read_bytes())
    require(meta['schema'] == SCHEMA and meta['algorithm'] == ALGORITHM
            and meta['hash_encoding'] == ENCODING, 'fresh record format')
    require(meta['torch_version'] == str(torch.__version__), 'fresh record Torch version differs')
    length, batch, seed, total = (meta[k] for k in ('length', 'batch_size', 'seed', 'total_updates'))
    require(all(type(v) is int for v in (length, batch, seed, total))
            and 0 < batch <= length and length % batch == 0 and total > 0
            and -(2**63) <= seed < 2**64, 'fresh record shape/seed')
    count = total if updates is None else updates
    require(type(count) is int and 0 < count <= total, 'fresh record update count')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    batches_per_epoch = length//batch
    epochs = (count+batches_per_epoch-1)//batches_per_epoch
    with (root/'epochs.jsonl').open('rb') as epoch_stream, (root/'samples.jsonl').open('rb') as batch_stream:
        update = 0
        for epoch in range(epochs):
            state = generator.get_state()
            order = torch.randperm(length, generator=generator)
            line = epoch_stream.readline()
            require(line.endswith(b'\n'), f'missing/torn epoch {epoch}')
            require(json.loads(line) == epoch_record(epoch, update, order, state), f'epoch {epoch} mismatch')
            for start in range(0, length, batch):
                if update == count:
                    break
                line = batch_stream.readline()
                require(line.endswith(b'\n'), f'missing/torn batch {update}')
                require(json.loads(line) == batch_record(update, epoch, order[start:start+batch]),
                        f'batch {update} mismatch')
                update += 1
        require(not epoch_stream.read(1) and not batch_stream.read(1), 'extra fresh records')
    return {'schema': SCHEMA, 'verified_updates': count, 'verified_epochs': epochs,
            'complete': count == total, 'seed': seed, 'length': length, 'batch_size': batch,
            'torch_version': str(torch.__version__)}
