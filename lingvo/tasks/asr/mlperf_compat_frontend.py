class MLPerfCompatFrontend(BaseAsrFrontend):
  @classmethod
  def Params(cls):
    p = super(MelAsrFrontend, cls).Params()
    p.name = 'frontend'
    p.Define('sample_rate', 16000.0, 'Sample rate in Hz')
    p.Define('channel_count', 1, 'Number of channels.')
    p.Define('frame_size_ms', 25.0,
             'Amount of data grabbed for each frame during analysis')
    p.Define('frame_step_ms', 10.0, 'Number of ms to jump between frames')
    p.Define('num_bins', 80, 'Number of bins in the mel-spectrogram output')
    p.Define('lower_edge_hertz', 125.0,
             'The lowest frequency of the mel-spectrogram analsis')
    p.Define('upper_edge_hertz', 7600.0,
             'The highest frequency of the mel-spectrogram analsis')
    p.Define('preemph', 0.97,
             'The first-order filter coefficient used for preemphasis')
    p.Define('noise_scale', 8.0,
             'The amount of noise (in 16-bit LSB units) to add')
    p.Define('window_fn', 'HANNING',
             'Window function to apply (valid values are "HANNING", and None)')
    p.Define(
        'pad_end', False,
        'Whether to pad the end of `signals` with zeros when the provided '
        'frame length and step produces a frame that lies partially past '
        'its end.')
    p.Define(
        'per_bin_mean', None,
        'Per-bin (num_bins) means for normalizing the spectrograms. '
        'Defaults to zeros.')
    p.Define('per_bin_stddev', None,
             'Per-bin (num_bins) standard deviations. Defaults to ones.')
    p.Define('stack_left_context', 0, 'Number of left context frames to stack.')
    p.Define('stack_right_context', 0,
             'Number of right context frames to stack.')
    p.Define('frame_stride', 1, 'The frame stride for sub-sampling.')

    return p

  @staticmethod
  def GetConfigFromParams(params):
    """Returns an AsrFrontendConfig namedtuple with vital config settings."""
    context_size = params.stack_left_context + params.stack_right_context + 1
    subsample_factor = params.num_bins * context_size
    frame_step = round(params.sample_rate * params.frame_step_ms / 1000.0)
    return AsrFrontendConfig(
        is_null=False,
        src_type='pcm',
        src_pcm_scale=32768.0,
        src_pcm_sample_rate=16000.0,
        output_dim=subsample_factor,
        input_frame_ratio=frame_step * subsample_factor)

  @base_layer.initializer
  def __init__(self, params):
    super(MelAsrFrontend, self).__init__(params)
    p = self.params
    if p.frame_stride < 1:
      raise ValueError('frame_stride must be positive.')

    assert p.channel_count == 1, 'Only 1 channel currently supported.'
    # Make sure key params are in floating point.
    p.sample_rate = float(p.sample_rate)
    p.frame_step_ms = float(p.frame_step_ms)
    p.frame_size_ms = float(p.frame_size_ms)
    p.lower_edge_hertz = float(p.lower_edge_hertz)
    p.upper_edge_hertz = float(p.upper_edge_hertz)

    self._frame_step = int(round(p.sample_rate * p.frame_step_ms / 1000.0))
    self._frame_size = (int(round(p.sample_rate * p.frame_size_ms / 1000.0)) + 1
                       )  # +1 for the preemph
    # Overdrive means double FFT size.
    # Note: 2* because of overdrive
    self._fft_size = 2 * int(max(512, _NextPowerOfTwo(self._frame_size)))

    self._CreateWindowFunction()

    # Mean/stddev.
    if p.per_bin_mean is None:
      p.per_bin_mean = [0.0] * p.num_bins
    if p.per_bin_stddev is None:
      p.per_bin_stddev = [1.0] * p.num_bins
    assert len(p.per_bin_mean) == p.num_bins
    assert len(p.per_bin_stddev) == p.num_bins

  def _CreateWindowFunction(self):
    p = self.params
    if p.window_fn is None:
      self._window_fn = None
    elif p.window_fn == 'HANNING':

      def _HanningWindow(frame_size, dtype):
        return tf.signal.hann_window(frame_size, dtype=dtype)

      self._window_fn = _HanningWindow
    else:
      raise ValueError('Illegal value %r for window_fn param' % (p.window_fn,))

  def FProp(self, theta, input_batch):
    """Perform signal processing on a sequence of PCM data.

    NOTE: This implementation does not currently support paddings, and they
    are accepted for compatibility with the super-class.

    TODO(laurenzo): Rework this to support paddings.

    Args:
      theta: Layer theta.
      input_batch: PCM input map:

        - 'src_inputs': int16 or float32 tensor of PCM audio data, scaled to
          +/-32768 (versus [-1..1)!). See class comments for supported input
          shapes.
        - 'paddings': per frame 0/1 paddings. Shaped: [batch, frame].
    Returns:
      NestedMap of encoder inputs which can be passed directly to a
      compatible encoder and contains:

        - 'src_inputs': inputs to the encoder, minimally of shape
          [batch, time, ...].
        - 'paddings': a 0/1 tensor of shape [batch, time].
    """

    return self._FPropDefault(input_batch)

  def _FPropDefault(self, input_batch):
    pcm_audio_data, pcm_audio_paddings = self._ReshapeToMono2D(
        input_batch.src_inputs, input_batch.paddings)

    mel_spectrogram, mel_spectrogram_paddings = self._FPropChunk(
        pcm_audio_data, pcm_audio_paddings)

    mel_spectrogram, mel_spectrogram_paddings = self._PadAndReshapeSpec(
        mel_spectrogram, mel_spectrogram_paddings)

    return py_utils.NestedMap(
        src_inputs=mel_spectrogram, paddings=mel_spectrogram_paddings)

  def _StackSignal(self, signal, stack_size, stride):
    signal = tf.signal.frame(
        signal=signal,
        frame_length=stack_size,
        frame_step=stride,
        pad_end=False,
        axis=1,
    )
    signal = tf.reshape(signal, py_utils.GetShape(signal)[:2] + [-1])
    return signal
