#include <rt/radar/scheduler.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>

PulseScheduler::PulseScheduler(std::size_t pri_steps, double dt, Waveform search_wave, Waveform track_wave) :
  pri_steps_(pri_steps),
  dt_(dt),
  search_wave_(search_wave),
  track_wave_(track_wave)
{
}

void PulseScheduler::on_step(std::size_t step, std::vector<PulseRequest>& out) {
  if (step < next_pulse_step_) return;
  next_pulse_step_ = step + pri_steps_;
  double fire_time = static_cast<double>(next_pulse_step_) * dt_;
  ++slot_;
  bool interleaved_track = track_interval_ > 0 && slot_ % track_interval_ == 0;
  bool with_track = !tracks_.empty() && (simultaneous_ || interleaved_track);
  bool with_search = simultaneous_ || !with_track;
  if (with_search) out.push_back(search_request(step, fire_time));
  if (with_track) out.push_back(track_request(step, fire_time));
}

void PulseScheduler::set_tracks(std::vector<Track> const& tracks) {
  tracks_.clear();
  for (auto const& track : tracks)
    if (track.confirmed) tracks_.push_back(track);
}

void PulseScheduler::set_track_interval(std::size_t interval) {
  assert(interval != 1);
  track_interval_ = interval;
}

std::size_t PulseScheduler::pulses_per_scan() const {
  auto search = static_cast<std::size_t>(std::lround(2.0 * abs_theta_ / theta_stride_));
  if (simultaneous_ || track_interval_ == 0) return search;
  return search * track_interval_ / (track_interval_ - 1);
}

PulseRequest PulseScheduler::search_request(std::size_t step, double fire_time) {
  next_theta_ += theta_stride_;
  if (next_theta_ >= abs_theta_) {
    next_theta_ = -abs_theta_;
  }
  return {step, search_wave_, next_theta_, fire_time, PulseMode::Search, 0};
}

PulseRequest PulseScheduler::track_request(std::size_t step, double fire_time) {
  auto const& track = tracks_[next_track_++ % tracks_.size()];
  double elapsed = fire_time - track.time;
  double out = track.out + track.v_out * elapsed;
  double along = track.along + track.v_along * elapsed;
  double theta = std::clamp(std::atan2(along, out), -abs_theta_, abs_theta_);
  return {step, track_wave_, theta, fire_time, PulseMode::Track, track.id};
}
