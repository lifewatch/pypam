LINEAR_UNITS = {'rms': ('L_rms', 'upa'),
                'dynamic_range': ('L_rms', 'upa'),
                'sel': ('SEL_rms', 'upa'),
                'peak': ('L_peak', 'upa'),
                'octave_levels': ('L_rms', 'upa'),
                'spectrogram_density': ('PSD', 'uPa^2/Hz'),
                'spectrogram_spectrum': ('PSD', 'uPa^2'),
                'spectrum_density': ('PSD', 'uPa^2/Hz'),
                'spectrum_spectrum': ('Spectrum level', 'uPa^2'),
                'spd': ('empirical_probability_density', '\%'),
                'waveform': ('amplitude', 'upa'),
                'aci': ('ACI', 'unitless'),
                'bi': ('BI', 'unitless'),
                'sh': ('SH', 'unitless'),
                'th': ('TH', 'unitless'),
                'ndsi': ('NDSI', 'unitless'),
                'aei': ('AEI', 'unitless'),
                'adi': ('ADI', 'unitless'),
                'zcr': ('zcr', 'unitless'),
                'zcr_avg': ('zcr_avg', 'unitless'),
                'kurtosis': ('Kurtosis', 'unitless'),
                }


def linear_units(method_name):
    return LINEAR_UNITS[method_name]


def logarithmic_units(method_name, p_ref):
    name, lin_unit = linear_units(method_name)
    log_unit = 'dB re %s %s' % (p_ref, lin_unit)
    return name, log_unit


def get_units(method_name, log, **kwargs):
    if log:
        if 'p_ref' not in kwargs.keys():
            raise ValueError('Units cannot be given in db without specifying the p_ref argument. '
                             'Error in method %s' % method_name)
        units = logarithmic_units(method_name, p_ref=kwargs['p_ref'])
    else:
        units = linear_units(method_name)

    return units


def get_units_attrs(method_name, **kwargs):
    name_units, units = get_units(method_name, **kwargs)
    return {'units': units, 'standard_name': name_units}
