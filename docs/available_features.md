# Available features

## Time domain features

In bioacoustics, sometimes several frequency bands are interesting to
study at the same time and extract features. pypam allows so by passing
a list of bands to analyze. Then, a features per \"time window\" and per
frequency band is computed.

For features analysis, the second dimension is:

- *bands*: all the bands analyzed (just as an int)

extra coordinates (metadata):

- *band_lowfreq*: low limit of the frequency band
- *band_highfreq*: high limit of the frequency band

The available features now are:

  ----------------------------------------------------------
  Feature name             Type             string
  ------------------------ ---------------- ----------------
  Acoustic Complexity      acoustic index   aci
  Index (ACI)                               

  Bioacoustic Index (BI)   acoustic index   bi

  Spectral Entropy of      acoustic index   sh
  Shannon (SH)                              

  Temporal Entropy of      acoustic index   th
  Shannon (SH)                              

  Normalized Difference    acoustic index   th
  Sound Index (NDSI)                        

  Acoustic Evenness Index  acoustic index   aei
  (AEI)                                     

  Acoustic Diversity Index acoustic index   adi
  (ADI)                                     

  Zero Crossing Rate (ZCR) acoustic index   zcr

  Acoustic Diversity Index acoustic index   adi
  (ADI)                                     

  Root Mean Squared Value  temporal feature rms
  (rms)                                     

  Sound Exposure Level     temporal feature sel
  (SEL)                                     

  Dynamic Range            temporal feature dynamic_range

  Peak                     temporal feature peak

  Root Mean Squared Value  temporal feature rms_envelope
  Envelope                                  

  Spectrum Slope           temporal feature spectrum_slope
  ----------------------------------------------------------

## Frequency domain features

[pypam]{.title-ref} allows to compute frequency-domain analysis
(spectrum, spectrogram, third octave band levels\...) per \"time
window\"

For frequency domain analysis, a second dimension, *frequency* is added.
This represents the central frequency of the frequency band

In case of spectrograms, a third dimension is added, *time*,
representing the seconds of the spectrogram at that specific time window

The available features are:

- spectrogram
- Spectrum
  - power_spectrum
  - psd (power spectral density)
- Spectral Probability Density (SPD)
- octave bands
