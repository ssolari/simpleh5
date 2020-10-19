
import blosc
import msgpack
import numpy as np
from typing import Any


def _encoder(obj):

    if isinstance(obj, np.ndarray):
        if obj.dtype == np.number or obj.dtype == np.int or obj.dtype == np.float or obj.dtype == np.bool:
            return {'data': obj.tobytes(), 'descr': obj.dtype.str, 'sh': obj.shape, 'isnp': True}
        else:
            # record arrays
            return {'data': obj.tobytes(), 'descr': obj.dtype.descr, 'isnp': True}
    elif isinstance(obj, np.number):
        return obj.item()

    raise TypeError


def _decoder(obj):

    try:
        if 'isnp' in obj:
            if 'sh' in obj:
                return np.frombuffer(obj['data'], dtype=obj['descr']).reshape(obj['sh'])
            else:
                return np.frombuffer(obj['data'], dtype=list(obj['descr']))
        else:
            return obj
    except KeyError:
        return obj


def msgpack_dumps(x, compress=True) -> bytes:
    # need to specially dump msgpack because it may be end padded with '\x00' which will get truncated
    data = msgpack.packb(x, use_bin_type=True, use_single_float=False, default=_encoder)
    if compress:
        return blosc.compress(data) + b'1'
    else:
        return data + b'1'


def msgpack_loads(x: bytes, use_list: bool = False, compress=True) -> Any:
    # need to specially read back msgpack minus last byte if dumped with extra byte
    if compress:
        data = blosc.decompress(x[:-1])
    else:
        data = x[:-1]
    return msgpack.unpackb(data, raw=False, use_list=use_list, strict_map_key=False, object_hook=_decoder)


def longest_obj_len(obj_list):
    """
    Utility function to determine the longest object len for determining dtype "O{len}"

    :param obj_list:
    :return:
    """
    long_len = 0
    for n in obj_list:
        packed_len = len(msgpack_dumps(n))
        if packed_len > long_len:
            long_len = packed_len
    return long_len


def longest_str_len(str_list):
    """
    Utility function to determine the longest string len for determining dtype "S{len}"

    :param obj_list:
    :return:
    """
    long_len = 0
    for n in str_list:
        n_len = len(n.encode('utf-8'))
        if n_len > long_len:
            long_len = n_len
    return long_len


def str_dtype(str_list):
    return "s{0}".format(longest_str_len(str_list))


def obj_dtype(str_list):
    return "o{0}".format(longest_obj_len(str_list))