.. currentmodule:: pypam

Available features
^^^^^^^^^^^^^^^^^^

Time domain features
~~~~~~~~~~~~~~~~~~~~
In bioacoustics, sometimes several frequency bands are interesting to study at the same time and extract features.
pypam allows so by passing a list of bands to analyze. Then, a features per "time window" and per frequency band is
computed.

For features analysis, the second dimension is:

* *bands*: all the bands analyzed (just as an int)

extra coordinates (metadata):

* *band_lowfreq*: low limit of the frequency band
* *band_highfreq*: high limit of the frequency band


The available features now are:

+------------------------+----------------+---------------+
| Feature name           | Type           | string        |
+========================+================+===============+
| Acoustic Complexity    | acoustic index | aci           |
| Index (ACI)            |                |               |
+------------------------+----------------+---------------+
| Bioacoustic Index (BI) | acoustic index | bi            |
+------------------------+----------------+---------------+
| Spectral Entropy of    | acoustic index | sh            |
| Shannon (SH)           |                |               |
+------------------------+----------------+---------------+
| Temporal Entropy of    | acoustic index | th            |
| Shannon (SH)           |                |               |
+------------------------+----------------+---------------+
| Normalized Difference  | acoustic index | th            |
| Sound Index (NDSI)     |                |               |
+------------------------+----------------+---------------+
| Acoustic Evenness      | acoustic index | aei           |
| Index (AEI)            |                |               |
+------------------------+----------------+---------------+
| Acoustic Diversity     | acoustic index | adi           |
| Index (ADI)            |                |               |
+------------------------+----------------+---------------+
| Zero Crossing Rate     | acoustic index | zcr           |
| (ZCR)                  |                |               |
+------------------------+----------------+---------------+
| Acoustic Diversity     | acoustic index | adi           |
| Index (ADI)            |                |               |
+------------------------+----------------+---------------+
| Root Mean Squared      | temporal       | rms           |
| Value (rms)            | feature        |               |
+------------------------+----------------+---------------+
| Sound Exposure Level   | temporal       | sel           |
| (SEL)                  | feature        |               |
+------------------------+----------------+---------------+
| Dynamic Range          | temporal       | dynamic_range |
|                        | feature        |               |
+------------------------+----------------+---------------+
| Peak                   | temporal       | peak          |
|                        | feature        |               |
+------------------------+----------------+---------------+
| Root Mean Squared      | temporal       | rms_envelope  |
| Value Envelope         | feature        |               |
+------------------------+----------------+---------------+
| Spectrum Slope         | temporal       | spectrum_slope|
|                        | feature        |               |
+------------------------+----------------+---------------+


Frequency domain features
~~~~~~~~~~~~~~~~~~~~~~~~~
`pypam` allows to compute frequency-domain analysis (spectrum, spectrogram, third octave band levels...) per "time window"

For frequency domain analysis, a second dimension, *frequency* is added. This represents the central frequency of the
frequency band

In case of spectrograms, a third dimension is added, *time*, representing the seconds of the spectrogram at that
specific time window

The available features are:

* spectrogram
* Spectrum
  * power_spectrum
  * psd (power spectral density)
* Spectral Probability Density (SPD)
* octave bands
