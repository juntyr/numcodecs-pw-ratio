import numcodecs
import numcodecs.abc
import numcodecs.compat
import numcodecs.registry
import numpy as np


def test_from_config():
    codec = numcodecs.registry.get_codec(
        dict(
            id="pw_ratio",
            eb_ratio=1.1,
            eb_abs_marker="$eb_abs",
            log_codec=dict(id="zlib", level=6),
            sign_codec=dict(id="zlib", level=6),
        )
    )
    assert codec.__class__.__name__ == "PointwiseRatioErrorBoundedCodec"
    assert codec.__class__.__module__ == "numcodecs_pw_ratio"

    assert (
        repr(codec)
        == "PointwiseRatioErrorBoundedCodec(eb_ratio=1.1, eb_abs_marker='$eb_abs', log_codec={'id': 'zlib', 'level': 6}, sign_codec=Zlib(level=6), pack_signs=True)"
    )


def check_roundtrip(data: np.ndarray, lossless_log_codec=False, pack_signs=True):
    for eb_ratio in [1.0, 1.1, 2.0, 10.0, 100.0]:
        codec = numcodecs.registry.get_codec(
            dict(
                id="pw_ratio",
                eb_ratio=eb_ratio,
                eb_abs_marker="$eb_abs",
                log_codec=dict(id="zlib")
                if lossless_log_codec
                else dict(id="sz3.rs", eb_mode="abs", eb_abs="$eb_abs"),
                sign_codec=dict(id="zlib"),
                pack_signs=pack_signs,
            )
        )

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert decoded.dtype == data.dtype
        assert decoded.shape == data.shape

        assert np.all(np.isnan(data) == np.isnan(decoded))
        assert np.all(data == decoded, where=np.isinf(data))
        assert np.all(np.signbit(data) == np.signbit(decoded))
        assert np.all((data == 0) == (decoded == 0))

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            assert np.all(
                np.abs(np.log2(np.abs(data)) - np.log2(np.abs(decoded)))
                <= np.log2(eb_ratio),
                where=(np.isfinite(data) & (data != 0)),
            )
            assert np.all(
                (data / decoded) <= eb_ratio, where=(np.isfinite(data) & (data != 0))
            )
            assert np.all(
                (decoded / data) <= eb_ratio, where=(np.isfinite(data) & (data != 0))
            )

        if eb_ratio == 1:
            assert np.all(data == decoded, where=(~np.isnan(data)))


def test_roundtrip():
    check_roundtrip(np.zeros(tuple()))
    check_roundtrip(np.zeros((0,)))
    check_roundtrip(np.arange(1000).reshape(10, 10, 10).astype(np.float64))
    check_roundtrip((np.arange(1000) - 500).reshape(10, 10, 10).astype(np.float64))
    check_roundtrip(np.sin(np.linspace(np.pi * -3, np.pi * 3, 1000)).astype(np.float32))
    check_roundtrip(np.array([4.2, -2.4, np.nan, -np.nan, 0.0, -0.0]))
    check_roundtrip(np.array([np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0]))
    check_roundtrip(
        np.array(
            [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0],
            dtype=np.dtype(np.float64).newbyteorder("<"),
        )
    )
    check_roundtrip(
        np.array(
            [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0],
            dtype=np.dtype(np.float64).newbyteorder(">"),
        ),
        lossless_log_codec=True,
    )


def test_unpacked_signs():
    data = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [0.0, -0.0, np.nan]])
    check_roundtrip(data, lossless_log_codec=True, pack_signs=False)
    check_roundtrip(data, lossless_log_codec=False, pack_signs=False)

    # the sign codec receives the boolean sign array of the data's shape
    class RecordingCodec(numcodecs.abc.Codec):
        codec_id = "recording-signs-test"

        def __init__(self):
            self.seen = None

        def encode(self, buf):
            self.seen = np.array(buf, copy=True)
            return self.seen.tobytes()

        def decode(self, buf, out=None):
            return numcodecs.compat.ndarray_copy(
                np.frombuffer(buf, dtype=self.seen.dtype).reshape(self.seen.shape), out
            )

    for pack_signs in (True, False):
        sign_codec = RecordingCodec()
        codec = numcodecs.registry.get_codec(
            dict(
                id="pw_ratio",
                eb_ratio=1.1,
                eb_abs_marker="$eb_abs",
                log_codec=dict(id="zlib"),
                sign_codec=sign_codec,
                pack_signs=pack_signs,
            )
        )
        decoded = codec.decode(codec.encode(data))
        assert np.all(np.signbit(data) == np.signbit(decoded))
        if pack_signs:
            assert sign_codec.seen.dtype == np.uint8 and sign_codec.seen.ndim == 1
        else:
            assert (
                sign_codec.seen.dtype == np.bool and sign_codec.seen.shape == data.shape
            )


def test_masked():
    class SparseLogCodec(numcodecs.abc.Codec):
        """Mask-aware test codec: stores only the unmasked values and records
        the masks it was given."""

        codec_id = "sparse-logs-test"

        def __init__(self):
            self.masks: list[np.ndarray] = []

        def encode(self, buf):
            return self.encode_masked(buf, np.zeros(np.shape(buf), dtype=np.bool))

        def decode(self, buf, out=None):
            return self.decode_masked(buf, np.zeros(np.shape(out), dtype=np.bool), out)

        def encode_masked(self, buf, mask):
            a = numcodecs.compat.ensure_ndarray(buf)
            assert mask.shape == a.shape and mask.dtype == np.bool
            self.masks.append(np.copy(mask))
            return a[~mask].tobytes()

        def decode_masked(self, buf, mask, out=None):
            assert out is not None and mask.shape == np.shape(out)
            decoded = np.zeros(mask.shape, dtype=np.asarray(out).dtype)
            decoded[~mask] = np.frombuffer(
                numcodecs.compat.ensure_bytes(buf), dtype=decoded.dtype
            )
            return numcodecs.compat.ndarray_copy(decoded, out)

        def get_config(self) -> dict:
            return dict(id=type(self).codec_id)

    data = np.array([[1.0, np.nan, 100.0], [-4.0, 1e-3, np.inf], [0.0, -25.0, 7.0]])
    mask = np.array([[False, True, False], [False, False, True], [False, False, False]])

    numcodecs.registry.register_codec(SparseLogCodec)

    codec = numcodecs.registry.get_codec(
        dict(
            id="pw_ratio",
            eb_ratio=1.1,
            eb_abs_marker="$eb_abs",
            log_codec=dict(id="sparse-logs-test"),
            sign_codec=dict(id="zlib", level=6),
        )
    )
    encoded = codec.encode_masked(data, mask)
    decoded = np.asarray(codec.decode_masked(encoded, mask)).reshape(data.shape)

    # the unmasked values obey the ratio error bound (NaN/inf excluded here)
    unmasked = ~mask
    finite = np.isfinite(data) & (data != 0) & unmasked
    ratio = decoded[finite] / data[finite]
    assert np.all((ratio >= 1 / 1.1) & (ratio <= 1.1))
    # zeros and signs of the unmasked values are preserved exactly
    assert np.all((decoded[unmasked & (data == 0)]) == 0)
    assert np.all(np.signbit(decoded[finite]) == np.signbit(data[finite]))
