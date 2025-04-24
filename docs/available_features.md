# Available features

The current available features are:

| Feature name                            | Type                      | string                   | Reference / Source code                                                                                 |
|-----------------------------------------|---------------------------|--------------------------|---------------------------------------------------------------------------------------------------------|
| Acoustic Complexity Index (ACI)         | acoustic index            | aci                      | [maad](https://scikit-maad.github.io/)                                                                  |
| Bioacoustic Index (BI)                  | acoustic index            | bi                       | [maad](https://scikit-maad.github.io/)                                                                  |
| Spectral Entropy of Shannon (SH)        | acoustic index            | sh                       | [maad](https://scikit-maad.github.io/)                                                                  |
| Temporal Entropy of Shannon (SH)        | acoustic index            | th                       | [maad](https://scikit-maad.github.io/)                                                                  |
| Acoustic Evenness Index (AEI)           | acoustic index            | aei                      | [maad](https://scikit-maad.github.io/)                                                                  |
| Acoustic Diversity Index (ADI)          | acoustic index            | adi                      | [maad](https://scikit-maad.github.io/)                                                                  |
| Zero Crossing Rate (ZCR)                | temporal feature          | zcr                      | https://github.com/patriceguyot/Acoustic_Indices                                                        |
| Root Mean Squared Value (RMS)           | temporal feature          | rms                      | ISO 18405:2017                                                                                          |
| Sound Exposure Level (SEL)              | temporal feature          | sel                      | ISO 18405:2017                                                                                          |
| Dynamic Range                           | temporal feature          | dynamic_range            | ISO 18405:2017                                                                                          |
| Cumulative Dynamic Range                | temporal feature          | cumulative_dynamic_range | ISO 18405:2017                                                                                          |
| Peak                                    | temporal feature          | peak                     | ISO 18405:2017                                                                                          |
| Spectrum Slope                          | temporal feature          | spectrum_slope           |                                                                                                         |
| Kurtosis                                | temporal feature          | kurtosis                 | [Muller et al. 2020](https://pubmed.ncbi.nlm.nih.gov/32872988/)                                         |
| Pulse width                             | temporal feature          | pulse_width              |                                                                                                         |
| Spectrogram                             | frequency domain feature  | spectrogram              | [scipy spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html) |
| Power spectrum                          | frequency domain feature  | power_spectrum           | ISO 18405:2017                                                                                          |
| Power spectral density                  | frequency domain feature  | psd                      | ISO 18405:2017                                                                                          |
| Spectral Probability Density (SPD)      | frequency domain feature  | spd                      | [Merchant et al. 2013](https://doi.org/10.1121/1.4794934)                                               |
| Octave bands (base 2) - filter bank     | frequency domain feature  | octaves_levels           |                                                                                                         |
| 1/3 Octave bands (base 2) - filter bank | frequency domain feature  | third_octaves_levels     |                                                                                                         |
| Hybrid millidecade band (base 10) - fft | frequency domain feature  | hybrid_millidecade_bands | [Martin et al. (2021)](https://doi.org/10.1121/10.0003324)                                              |

Hybrid millidecade bands also can be converted to decidecade bands using the function utils.hmb_to_decidecade(). 
This will be implemented as a feature in the short future. 

Long Term Spectral Average (LTSA) is computed by computing the spectrum of a full deployment. 

## Resulting coordinates

### Temporal Features in different bands
In bioacoustics, sometimes several frequency bands are interesting to study at the same time and extract features. 
*pypam* allows so by passing a list of bands to analyze. 
Features are computed per "time window" and frequency band.

For features analysis, the second dimension is:

- *bands*: all the bands analyzed (just as an int)

extra coordinates (metadata):

- *band_lowfreq*: low limit of the frequency band
- *band_highfreq*: high limit of the frequency band


### Frequency-domain features
For frequency domain analysis, a second dimension is added: 

- *frequency*, representing the central frequency of the frequency band


In case of spectrograms, a third dimension is added: 
- *time*, representing the seconds of the spectrogram at that specific time window

In case of hybrid millidecade bands, *frequency* is used for the computed FFT spectra. For the hybrid millidecade bands 
themselves, a new dimension is added:

- *frequency_bins*, representing the central frequency of the frequency band

And its two corresponding metadata coordinates: 

- *upper_frequency*, representing the upper bound of the frequency band
- *lower_frequency*, representing the lower bound of the frequency band


