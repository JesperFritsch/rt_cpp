#pragma once
#include <rt/radar/types.hpp>

// linear chirp from f_start to f_end; the phase is the accumulated frequency,
// so it grows as f_start * t + 0.5 * chirp_rate * t^2 rather than f(t) * t
// `phase` shifts the carrier, pi / 2 gives the quadrature copy the matched filter uses
float wave_contrib(double time_in_pulse, Waveform const& wave, double phase = 0.0);
