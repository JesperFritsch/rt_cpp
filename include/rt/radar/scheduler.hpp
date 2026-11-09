#pragma once
#include <rt/radar/types.hpp>
#include <vector>

class PulseScheduler {
public:
  PulseScheduler(std::size_t pri_steps, double dt, Waveform search_wave, Waveform track_wave);

  // appends the requests for `step` to out: none between pulses, otherwise a Search pulse, a Track
  // pulse, or both fired at the same step; a Search pulse comes first so its scan bookkeeping does too
  void on_step(std::size_t step, std::vector<PulseRequest>& out);

  void set_simultaneous_tracks(bool simultaneous) { simultaneous_ = simultaneous; }

  // the tracks to aim Track pulses at; only confirmed ones are kept
  void set_tracks(std::vector<Track> const& tracks);

  // 1 would make every pulse a Track pulse whenever there are tracks, and the scan would stop
  void set_track_interval(std::size_t interval);

  void set_angles(double abs_angle_r, double ang_stride_r) {
    abs_theta_ = abs_angle_r;
    theta_stride_ = ang_stride_r;
  }

  // angles from -abs_theta_ up to, but not including, +abs_theta_, plus the Track pulses
  // between them when there are tracks to aim at
  std::size_t pulses_per_scan() const;

private:
  PulseRequest search_request(std::size_t step, double fire_time);

  // the confirmed tracks take turns; aim where the track expects the target at fire_time
  PulseRequest track_request(std::size_t step, double fire_time);

  double theta_stride_;
  double abs_theta_;
  std::size_t pri_steps_;
  double dt_;
  Waveform search_wave_;
  Waveform track_wave_;
  std::size_t next_pulse_step_ = 0;
  double next_theta_ = 0.0;
  std::size_t slot_ = 0;            // pulses scheduled so far
  std::size_t track_interval_ = 0;
  bool simultaneous_ = false;
  std::vector<Track> tracks_;       // confirmed tracks, as of the latest set_tracks
  std::size_t next_track_ = 0;      // round robin over tracks_
};
