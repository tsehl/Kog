"""Builds a read-only extension with the headers and the compiler of the local oracle."""
from pathlib import Path
import ctypes
import fcntl
import functools
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CORE = HERE / 'ygopro-core'
OUT = HERE.parent / '.tools'


@functools.lru_cache(maxsize=1)
def load():
    OUT.mkdir(exist_ok=True)
    binary = OUT / 'libeffect-inspector.so'
    stamp = OUT / 'effect-inspector.json'
    compiler = os.environ.get('CXX', 'g++')
    inputs = [HERE / 'effect_inspector.cpp', CORE / ('bin/release/libocgcore.dylib' if sys.platform == 'darwin' else 'bin/release/libocgcore.so'), Path(__file__)]
    inputs += sorted(CORE.rglob('*.h')) + sorted(CORE.rglob('*.hpp'))
    signature = hashlib.sha256(compiler.encode() + b''.join(p.read_bytes() for p in inputs) +
                               subprocess.check_output([compiler, '--version'])).hexdigest()
    with (OUT / 'effect-inspector.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            cached = json.loads(stamp.read_text())
        except (OSError, ValueError):
            cached = {}
        if not binary.exists() or not isinstance(cached, dict) or cached.get('signature') != signature:
            temporary = binary.with_suffix('.tmp.so')
            subprocess.run([compiler, '-std=c++17', '-O2', ('-dynamiclib' if sys.platform == 'darwin' else '-shared'), '-fPIC', '-fno-rtti',
                            '-I'+str(CORE), '-I'+str(CORE/'lua/src'), str(inputs[0]), '-o', str(temporary)], check=True)
            os.replace(temporary, binary)
            stamp_tmp = stamp.with_suffix('.tmp')
            stamp_tmp.write_text(json.dumps(dict(signature=signature)))
            os.replace(stamp_tmp, stamp)
    lib = ctypes.CDLL(str(binary))
    lib.YGO_EffectEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
    lib.YGO_EffectEvent.restype = ctypes.c_int
    lib.YGO_EffectDescription.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint),
                                          ctypes.POINTER(ctypes.c_uint64)]
    lib.YGO_EffectDescription.restype = ctypes.c_int
    for name in ('YGO_CardIds', 'YGO_OverlayIds'):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_uint]
        fn.restype = ctypes.c_int
    return lib


def card_ids(duel, player, location):
    """cardid of the cards of an oracle location (None for an empty zone), see effect_inspector.cpp."""
    buf = (ctypes.c_uint * 64)()
    n = load().YGO_CardIds(duel, player, location, buf, 64)
    return [None if buf[i] == 0xFFFFFFFF else buf[i] for i in range(n)]


def overlay_ids(duel, player, seq):
    buf = (ctypes.c_uint * 16)()
    n = load().YGO_OverlayIds(duel, player, seq, buf, 16)
    return [buf[i] for i in range(n)]


def event(duel, index, expected_code):
    code, value = ctypes.c_uint(), ctypes.c_uint()
    if not load().YGO_EffectEvent(duel, index, ctypes.byref(code), ctypes.byref(value)):
        raise RuntimeError('effect identity unavailable on the oracle side')
    if code.value != expected_code:
        raise RuntimeError(f'inconsistent oracle effect identity: {code.value} / {expected_code}')
    return value.value


def description(duel, index, expected_code):
    code, value = ctypes.c_uint(), ctypes.c_uint64()
    if not load().YGO_EffectDescription(duel, index, ctypes.byref(code), ctypes.byref(value)):
        raise RuntimeError('effect description unavailable on the oracle side')
    if code.value != expected_code:
        raise RuntimeError(f'inconsistent oracle effect identity: {code.value} / {expected_code}')
    return value.value


if __name__ == '__main__':
    load()
    print('Oracle effect inspector built and loaded')
