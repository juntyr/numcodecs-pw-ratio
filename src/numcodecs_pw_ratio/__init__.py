"""
[`PointwiseRatioErrorBoundedCodec`][numcodecs_pw_ratio.PointwiseRatioErrorBoundedCodec] meta-codec for the [`numcodecs`][numcodecs] buffer compression API, which preserves a ratio/logarithmic error bound using an absolute-error-bounded codec.
"""

__all__ = ["PointwiseRatioErrorBoundedCodec"]

import copy
from functools import reduce
from io import BytesIO
from typing import Callable, ClassVar, Optional, overload

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
from typing_extensions import Buffer  # MSPV 3.12


class PointwiseRatioErrorBoundedCodec(Codec, CodecCombinatorMixin):
    r"""
    Meta-codec that preserves a ratio error bound by wrapping an absolute-error-bounded codec.

    The ratio error bound `eb_ratio` is translated into an absolute error bound
    for the `log_codec`, which is used to encode the logarithms of the data.
    The meta-codec preserves infinite and NaN values, if the wrapped `log_codec`
    preserves them, and supports all positive, zero, and negative floating-point
    values. The `sign_codec` is used to to losslessly encode the signs of the
    data.

    The `log_codec` configuration should include a marker, `eb_abs_marker`,
    which is replaced with the translated absolute error bound.

    The implementation of the meta-codec is based on Ling et al.[^1] and adopted
    from `libpressio`[^2]'s `pw_rel_compressor_plugin`[^3] meta-compressor
    plugin.

    A ratio error bound guarantees that the ratios between the original and the
    decoded values as well as their inverse ratios are less than or equal to
    the provided bound $\epsilon_{ratio}$:

    \[
        \left\{\begin{array}{lr}
            0 \quad &\text{if } x = \hat{x} = 0 \\
            \inf \quad &\text{if } \text{sign}(x) \neq \text{sign}(\hat{x}) \\
            |\log(|x|) - \log(|\hat{x}|)| \quad &\text{otherwise}
        \end{array}\right\} \leq \log(\epsilon_{ratio})
    \]

    for a finite $\epsilon_{ratio} \geq 1$.

    A ratio error bound also guarantees that the sign of each decoded value
    matches the sign of each original value and that a decoded value is zero if
    and only if it is zero in the original data.

    The ratio error bound is sometimes also known as a decimal error bound[^4]
    [^5] if the ratio is expressed as the difference in orders of magnitude. A
    decimal error bound of e.g. $2$ (two orders of magnitude difference / x100
    ratio) can be expressed using
    $\epsilon_{ratio} = {10}^{\epsilon_{decimal}}$.

    [^1]: Liang, X., Di, S., Tao, D., Chen, Z., & Cappello, F. (2018). An
        Efficient Transformation Scheme for Lossy Data Compression with
        Point-Wise Relative Error Bound. *2018 IEEE International Conference on
        Cluster Computing (CLUSTER)*, 179–189. Available from:
        [doi:10.1109/cluster.2018.00036](https://doi.org/10.1109/cluster.2018.00036).

    [^2]: Underwood, R., Malvoso, V., Calhoun, J. C., Di, S., & Cappello, F.
        (2021). Productive and Performant Generic Lossy Data Compression with
        LibPressio. *2021 7th International Workshop on Data Analysis and
        Reduction for Big Scientific Data (DRBSD-7)*, 1–10. Available from:
        [doi:10.1109/drbsd754563.2021.00005](https://doi.org/10.1109/drbsd754563.2021.00005).

    [^3]: <https://github.com/robertu94/libpressio/blob/868a3a70d6ebf55ad67509fbca03bdd0bc1bc246/src/plugins/compressors/pw_rel.cc>

    [^4]: Gustafson, J. L., & Yonemoto, I. T. (2017). Beating floating-point at
        its Own Game: Posit Arithmetic. *Supercomputing Frontiers and
        Innovations*, 4(2). Available from:
        [doi:10.14529/jsfi170206](https://doi.org/10.14529/jsfi170206).

    [^5]: Klöwer, M., Düben, P. D., & Palmer, T. N. (2019). Posits as an
        alternative to floats for weather and climate models. *CoNGA'19:
        Proceedings of the Conference for Next Generation Arithmetic 2019*, 1-8.
        Available from:
        [doi:10.1145/3316279.3316281](https://doi.org/10.1145/3316279.3316281).

    Parameters
    ----------
    eb_ratio : float
        The finite ratio error bound, $\geq 1$.
    eb_abs_marker : str
        The marker for the absolute error bound in the `log_codec`.
    log_codec : dict
        The configuration for the absolute-error-bounded codec that encodes
        the logarithms of the data.
    sign_codec : dict | Codec
        The configuration or instantiated codec that encodes the data signs.
    """

    __slots__: tuple[str, ...] = (
        "_eb_ratio",
        "_eb_abs_marker",
        "_log_codec",
        "_sign_codec",
        "_mapper",
    )
    _eb_ratio: float
    _eb_abs_marker: str
    _log_codec: dict
    _sign_codec: Codec
    _mapper: Callable[[Codec], Codec]

    codec_id: ClassVar[str] = "pw_ratio"  # type: ignore

    def __init__(
        self,
        eb_ratio: float,
        eb_abs_marker: str,
        log_codec: dict,
        sign_codec: dict | Codec,
    ):
        if eb_ratio < 1:
            raise ValueError("eb_ratio must be >= 1")

        if not np.isfinite(eb_ratio):
            raise ValueError("eb_ratio must be finite")

        self._eb_ratio = eb_ratio
        self._eb_abs_marker = eb_abs_marker
        self._log_codec = copy.deepcopy(log_codec)
        self._sign_codec = (
            sign_codec
            if isinstance(sign_codec, Codec)
            else numcodecs.registry.get_codec(sign_codec)
        )
        self._mapper = lambda x: x

    def encode(self, buf: Buffer) -> bytes:  # type: ignore
        """Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        if not np.issubdtype(dtype, np.floating):
            raise TypeError("can only encode floating point values")

        signs = np.signbit(a)
        save_signs = np.any(signs)

        logs = np.copy(a)
        np.negative(logs, out=logs, where=signs, casting="equiv")

        valid_logs = np.isfinite(logs) & (logs > 0)

        np.log2(logs, out=logs, where=valid_logs, casting="equiv")

        max_log_data = np.amax(logs, where=valid_logs, initial=-np.inf)
        min_log_data = np.amin(logs, where=valid_logs, initial=np.inf)

        max_abs_log_data = np.maximum(np.abs(min_log_data), np.abs(max_log_data))

        eb_abs = np.log2(self._eb_ratio) - max_abs_log_data * np.finfo(dtype).eps

        if not (np.isfinite(eb_abs) and eb_abs > 0.0):
            eb_abs = np.array(eb_abs).dtype.type(0)

        threshold = np.minimum(
            np.nextafter(min_log_data, dtype.type(-np.inf)),
            min_log_data - 1.0001 * eb_abs,
        )
        zero_flag = np.minimum(
            np.nextafter(threshold, dtype.type(-np.inf)), min_log_data - 2.0001 * eb_abs
        )

        logs[a == 0] = zero_flag

        logs_codec: Codec = numcodecs.registry.get_codec(
            _replace_eb_abs_in_config(
                self._log_codec,
                self._eb_abs_marker,
                float(eb_abs) if isinstance(eb_abs, float) else eb_abs,
            )
        )

        logs_encoded = logs_codec.encode(logs if eb_abs > 0 else a)
        logs_encoded = numcodecs.compat.ensure_ndarray(logs_encoded)

        if save_signs and (eb_abs > 0):
            signs_encoded = self._sign_codec.encode(
                np.packbits(signs, axis=None, bitorder="big")
            )
            signs_encoded = numcodecs.compat.ensure_ndarray(signs_encoded)
        else:
            signs_encoded = None

        # message: dtype shape eb_abs_dtype eb_abs ...a
        #  (eb_abs == 0) => ...b
        #  (eb_abs >  0) => a... threshold_dtype threshold ...b
        #         (True) => b... logs_encoded_dtype logs_encoded_shape [padding]
        #                   logs_encoded ...c
        #  (eb_abs == 0) => c... EOF
        #  (!save_signs) => c... 0 EOF
        #   (save_signs) => c... signs_dtype signs_shape [padding] signs_encoded
        #                   EOF
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        eb_abs = np.array(eb_abs)
        message.append(leb128.u.encode(len(eb_abs.dtype.str)))
        message.append(eb_abs.dtype.str.encode("ascii"))

        # ensure that the encoded eb_abs value is encoded in little endian binary
        message.append(eb_abs.astype(eb_abs.dtype.newbyteorder("<")).tobytes())

        if eb_abs > 0:
            threshold = np.array(threshold)
            message.append(leb128.u.encode(len(threshold.dtype.str)))
            message.append(threshold.dtype.str.encode("ascii"))

            # ensure that the encoded threshold value is encoded in little endian binary
            message.append(
                threshold.astype(threshold.dtype.newbyteorder("<")).tobytes()
            )

        message.append(leb128.u.encode(len(logs_encoded.dtype.str)))
        message.append(logs_encoded.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(logs_encoded.ndim))
        for s in logs_encoded.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with logs itemsize
        message.append(
            b"\0"
            * (
                logs_encoded.dtype.itemsize
                - (sum(len(m) for m in message) % logs_encoded.itemsize)
            )
        )

        # ensure that the encoded logs values are encoded in little endian binary
        message.append(
            logs_encoded.astype(logs_encoded.dtype.newbyteorder("<")).tobytes()
        )

        if eb_abs > 0:
            if signs_encoded is None:
                message.append(leb128.u.encode(0))
            else:
                message.append(leb128.u.encode(len(signs_encoded.dtype.str)))
                message.append(signs_encoded.dtype.str.encode("ascii"))

                message.append(leb128.u.encode(signs_encoded.ndim))
                for s in signs_encoded.shape:
                    message.append(leb128.u.encode(s))

                # insert padding to align with signs itemsize
                message.append(
                    b"\0"
                    * (
                        signs_encoded.dtype.itemsize
                        - (sum(len(m) for m in message) % signs_encoded.itemsize)
                    )
                )

                # ensure that the encoded signs values are encoded in little endian binary
                message.append(
                    signs_encoded.astype(
                        signs_encoded.dtype.newbyteorder("<")
                    ).tobytes()
                )

        return b"".join(message)

    def decode(self, buf: Buffer, out: Optional[Buffer] = None) -> Buffer:  # type: ignore
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        b = numcodecs.compat.ensure_bytes(buf)
        b_io = BytesIO(b)

        # message: dtype shape eb_abs_dtype eb_abs ...a
        #  (eb_abs == 0) => ...b
        #  (eb_abs >  0) => a... threshold_dtype threshold ...b
        #         (True) => b... logs_encoded_dtype logs_encoded_shape [padding]
        #                   logs_encoded ...c
        #  (eb_abs == 0) => c... EOF
        #  (!save_signs) => c... 0 EOF
        #   (save_signs) => c... signs_dtype signs_shape [padding] signs_encoded
        #                   EOF

        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        size = reduce(lambda a, b: a * b, shape, 1)

        eb_abs_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        eb_abs = np.frombuffer(
            b_io.read(eb_abs_dtype.itemsize),
            dtype=eb_abs_dtype.newbyteorder("<"),
            count=1,
        ).astype(eb_abs_dtype)[0]

        if eb_abs > 0:
            threshold_dtype = np.dtype(
                b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
            )
            threshold = np.frombuffer(
                b_io.read(threshold_dtype.itemsize),
                dtype=threshold_dtype.newbyteorder("<"),
                count=1,
            ).astype(threshold_dtype)[0]
        else:
            threshold = None

        logs_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        logs_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        logs_size = reduce(lambda a, b: a * b, logs_shape, 1)

        # remove padding to align with logs itemsize
        b_io.read(logs_dtype.itemsize - (b_io.tell() % logs_dtype.itemsize))

        logs_encoded = (
            np.frombuffer(
                b_io.read(logs_size * logs_dtype.itemsize),
                dtype=logs_dtype.newbyteorder("<"),
                count=logs_size,
            )
            .astype(logs_dtype)
            .reshape(logs_shape)
        )

        if eb_abs == 0:
            signs_encoded = None
        else:
            signs_dtype_len = leb128.u.decode_reader(b_io)[0]

            if signs_dtype_len == 0:
                signs_encoded = None
            else:
                signs_dtype = np.dtype(b_io.read(signs_dtype_len).decode("ascii"))
                signs_shape = tuple(
                    leb128.u.decode_reader(b_io)[0]
                    for _ in range(leb128.u.decode_reader(b_io)[0])
                )
                signs_size = reduce(lambda a, b: a * b, signs_shape, 1)

                # remove padding to align with signs itemsize
                b_io.read(signs_dtype.itemsize - (b_io.tell() % signs_dtype.itemsize))

                signs_encoded = (
                    np.frombuffer(
                        b_io.read(signs_size * signs_dtype.itemsize),
                        dtype=signs_dtype.newbyteorder("<"),
                        count=signs_size,
                    )
                    .astype(signs_dtype)
                    .reshape(signs_shape)
                )

        logs = np.empty(shape, dtype=dtype)

        logs_codec: Codec = numcodecs.registry.get_codec(
            _replace_eb_abs_in_config(
                self._log_codec,
                self._eb_abs_marker,
                float(eb_abs) if isinstance(eb_abs, float) else eb_abs,  # type: ignore
            )
        )
        logs_codec.decode(logs_encoded, out=logs)

        if threshold is not None:
            np.exp2(
                logs,
                out=logs,
                where=(np.isfinite(logs) & (logs >= threshold)),
                casting="equiv",
            )
            logs[np.isfinite(logs) & (logs < threshold)] = 0

        if signs_encoded is not None:
            packed_signs = np.empty((size + 7) // 8, dtype=np.uint8)

            self._sign_codec.decode(signs_encoded, out=packed_signs)

            signs = (
                np.unpackbits(packed_signs, axis=None, count=size, bitorder="big")
                .astype(np.bool)
                .reshape(shape)
            )

            np.negative(logs, out=logs, where=signs, casting="equiv")

        return numcodecs.compat.ndarray_copy(logs, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this pointwise ratio error bounded codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this pointwise ratio error bounded codec.
        """

        return dict(
            id=type(self).codec_id,
            eb_ratio=self._eb_ratio,
            eb_abs_marker=self._eb_abs_marker,
            log_codec=copy.deepcopy(self._log_codec),
            sign_codec=self._sign_codec.get_config(),
        )

    def map(
        self, mapper: Callable[[Codec], Codec]
    ) -> "PointwiseRatioErrorBoundedCodec":
        """
        Apply the `mapper` to this pointwise ratio error bounded codec.

        In the returned [`PointwiseRatioErrorBoundedCodec`][..], the
        `log_codec` and `sign_codec` are replaced by their mapped codecs.

        The `mapper` should recursively apply itself to any inner codecs that
        also implement the
        [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin]
        mixin.

        To automatically handle the recursive application as a caller, you can
        use
        ```python
        numcodecs_combinators.map_codec(codec, mapper)
        ```
        instead.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that should be applied to the wrapped `log_codec` and
            `sign_codec` to map over this pointwise ratio error bounded codec.

        Returns
        -------
        mapped : PointwiseRatioErrorBoundedCodec
            The mapped pointwise ratio error bounded codec.
        """

        codec = PointwiseRatioErrorBoundedCodec(
            self._eb_ratio,
            self._eb_abs_marker,
            copy.deepcopy(self._log_codec),
            mapper(self._sign_codec),
        )

        old_mapper = self._mapper
        codec._mapper = lambda x: mapper(old_mapper(x))

        return codec


def _replace_eb_abs_in_config(config: dict, eb_abs_marker: str, eb_abs: float) -> dict:
    @overload
    def _replace_in(x: dict) -> dict: ...

    @overload
    def _replace_in(x: list) -> list: ...

    @overload
    def _replace_in(x: tuple) -> tuple: ...

    def _replace_in(x):
        if isinstance(x, dict):
            return {
                k: (eb_abs if v == eb_abs_marker else _replace_in(v))
                for k, v in x.items()
            }
        if isinstance(x, list | tuple):
            return type(x)(_replace_in(xi) for xi in x)
        return x

    return _replace_in(config)
