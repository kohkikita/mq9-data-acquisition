# daq_app/audio.py
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    def __init__(self, wav_path: str, fs: int, channels: int, device=None):
        self.wav_path = wav_path
        self.fs = fs
        self.channels = channels
        self.device = device
        self._sf = None
        self._stream = None

    def start(self):
        self._sf = sf.SoundFile(
            self.wav_path, mode="w", samplerate=self.fs, channels=self.channels, subtype="PCM_16"
        )

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            self._sf.write(indata)

        self._stream = sd.InputStream(
            samplerate=self.fs,
            channels=self.channels,
            device=self.device,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def stop(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            if self._sf is not None:
                self._sf.close()
                self._sf = None
