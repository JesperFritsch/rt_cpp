#include <rt/radar/waveform.hpp>

float wave_contrib(double time_in_pulse, Waveform const& wave, double phase) {
  double chirp_rate = (wave.f_end - wave.f_start) / wave.duration;  // Hz per second
  double cycles = wave.f_start * time_in_pulse + 0.5 * chirp_rate * time_in_pulse * time_in_pulse;
  // a half cosine ramp over `taper` of the duration at each end (a Tukey window) instead of switching
  // on and off at once, which keeps the spectrum much closer to the chirp's own band
  double envelope = 1.0;
  double ramp = wave.taper * wave.duration;
  double from_edge = std::min(time_in_pulse, wave.duration - time_in_pulse);
  if (ramp > 0.0 && from_edge < ramp) envelope = 0.5 * (1.0 - std::cos(pi * from_edge / ramp));
  auto contrib = time_in_pulse < wave.duration ? envelope * wave.amplitude * std::sin(2.0 * pi * cycles + phase) : 0.0;
  return static_cast<float>(contrib);
}
