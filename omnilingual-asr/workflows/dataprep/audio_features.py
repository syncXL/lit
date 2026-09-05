import io
import numpy as np
import pyarrow as pa
import soundfile as sf


class AudioFeatureProcessor:
    """
    Extract acoustic features from each audio utterance.

    Features:
        snr_db
        noise_level_db
        speech_level_db
        speech_ratio
        duration_sec
        sample_rate
        channels
        spectral_bandwidth_hz

    SNR/noise/speech levels are estimated from frame-level RMS energy.
    The lowest-energy frames are treated as the estimated noise floor.
    """

    def __init__(
        self,
        audio_column: str = "audio",
        frame_ms: int = 25,
        hop_ms: int = 10,
        noise_percentile: float = 10.0,
    ):
        self.audio_column = audio_column
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.noise_percentile = noise_percentile

    @staticmethod
    def _db(rms: float) -> float:
        return float(20.0 * np.log10(max(rms, 1e-10)))

    def _extract_one(self, audio):
        if isinstance(audio, dict):
            audio = audio.get("bytes")

        if audio is None:
            return self._empty_features()

        try:
            # Decode audio
            samples, sr = sf.read(
                io.BytesIO(audio),
                dtype="float32",
            )

            # Convert to mono for acoustic analysis
            if samples.ndim == 1:
                channels = 1
                mono = samples
            else:
                channels = samples.shape[1]
                mono = samples.mean(axis=1)

            n_samples = len(mono)

            # Duration
            duration_sec = n_samples / sr

            # Frame configuration
            frame_len = max(
                1,
                int(sr * self.frame_ms / 1000),
            )

            hop_len = max(
                1,
                int(sr * self.hop_ms / 1000),
            )

            # Create overlapping frames
            if n_samples < frame_len:
                frames = mono[None, :]
            else:
                frames = np.lib.stride_tricks.sliding_window_view(
                    mono,
                    frame_len,
                )[::hop_len]

            # ---------------------------------------------------------
            # RMS energy
            # ---------------------------------------------------------

            rms = np.sqrt(
                np.mean(frames ** 2, axis=1) + 1e-12
            )

            # Lowest-energy frames are treated as noise.
            noise_threshold = np.percentile(
                rms,
                self.noise_percentile,
            )

            noise_frames = rms <= noise_threshold
            speech_frames = rms > noise_threshold

            # Safety fallback
            if not np.any(noise_frames):
                noise_frames = np.ones_like(
                    rms,
                    dtype=bool,
                )

            if not np.any(speech_frames):
                speech_frames = np.ones_like(
                    rms,
                    dtype=bool,
                )

            # ---------------------------------------------------------
            # Noise level
            # ---------------------------------------------------------

            noise_rms = float(
                np.sqrt(
                    np.mean(
                        rms[noise_frames] ** 2
                    )
                )
            )

            noise_level_db = self._db(noise_rms)

            # ---------------------------------------------------------
            # Speech level
            # ---------------------------------------------------------

            speech_rms = float(
                np.sqrt(
                    np.mean(
                        rms[speech_frames] ** 2
                    )
                )
            )

            speech_level_db = self._db(speech_rms)

            # ---------------------------------------------------------
            # SNR
            # ---------------------------------------------------------

            snr_db = speech_level_db - noise_level_db

            # ---------------------------------------------------------
            # Speech ratio
            # ---------------------------------------------------------

            speech_ratio = float(
                np.mean(speech_frames)
            )

            # ---------------------------------------------------------
            # Spectral bandwidth
            # ---------------------------------------------------------

            window = np.hanning(frame_len)

            bandwidths = []

            for frame in frames:
                # Pad short final frame
                if len(frame) < frame_len:
                    frame = np.pad(
                        frame,
                        (0, frame_len - len(frame)),
                    )

                spectrum = np.abs(
                    np.fft.rfft(
                        frame * window
                    )
                )

                frequencies = np.fft.rfftfreq(
                    frame_len,
                    1 / sr,
                )

                magnitude_sum = spectrum.sum()

                if magnitude_sum <= 0:
                    continue

                # Spectral centroid
                centroid = (
                    np.sum(
                        frequencies * spectrum
                    )
                    / magnitude_sum
                )

                # Spectral bandwidth
                bandwidth = np.sqrt(
                    np.sum(
                        (
                            (frequencies - centroid)
                            ** 2
                        )
                        * spectrum
                    )
                    / magnitude_sum
                )

                bandwidths.append(
                    float(bandwidth)
                )

            spectral_bandwidth_hz = (
                float(np.mean(bandwidths))
                if bandwidths
                else 0.0
            )

            return {
                "snr_db": snr_db,
                "noise_level_db": noise_level_db,
                "speech_level_db": speech_level_db,
                "speech_ratio": speech_ratio,
                "duration_sec": duration_sec,
                "sample_rate": sr,
                "channels": channels,
                "spectral_bandwidth_hz": spectral_bandwidth_hz,
            }

        except Exception:
            return self._empty_features()

    @staticmethod
    def _empty_features():
        return {
            "snr_db": None,
            "noise_level_db": None,
            "speech_level_db": None,
            "speech_ratio": None,
            "duration_sec": None,
            "sample_rate": None,
            "channels": None,
            "spectral_bandwidth_hz": None,
        }

    def __call__(self, batch: pa.Table) -> pa.Table:
        audio = batch[self.audio_column].to_pylist()

        features = [
            self._extract_one(item)
            for item in audio
        ]

        feature_types = {
            "snr_db": pa.float32(),
            "noise_level_db": pa.float32(),
            "speech_level_db": pa.float32(),
            "speech_ratio": pa.float32(),
            "duration_sec": pa.float32(),
            "sample_rate": pa.int32(),
            "channels": pa.int32(),
            "spectral_bandwidth_hz": pa.float32(),
        }

        for name, dtype in feature_types.items():
            values = [
                item[name]
                for item in features
            ]

            batch = batch.append_column(
                name,
                pa.array(
                    values,
                    type=dtype,
                ),
            )

        return batch