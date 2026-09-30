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
