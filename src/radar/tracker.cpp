#include <rt/radar/tracker.hpp>
#include <algorithm>
#include <optional>

Tracker::Tracker(RadarConfig const& config) : config_(config) {
  tracks_.reserve(32);
  pairs_.reserve(256);
  plot_taken_.reserve(32);
  track_corrected_.reserve(32);
}

void Tracker::update(std::vector<Plot> const& plots) {
  // every track and plot pair within the gate, closest first; a track or plot is used at most once
  pairs_.clear();
  for (std::size_t t = 0; t < tracks_.size(); ++t) {
    for (std::size_t p = 0; p < plots.size(); ++p) {
      auto [out, along] = predict(tracks_[t], plots[p].time);
      double distance = std::hypot(out_of(plots[p]) - out, along_of(plots[p]) - along);
      if (distance <= config_.track_gate) pairs_.push_back({distance, t, p});
    }
  }
  std::sort(pairs_.begin(), pairs_.end(), [](Pair const& a, Pair const& b) { return a.distance < b.distance; });

  plot_taken_.assign(plots.size(), false);
  track_corrected_.assign(tracks_.size(), false);
  for (auto const& pair : pairs_) {
    if (plot_taken_[pair.plot] || track_corrected_[pair.track]) continue;
    correct(tracks_[pair.track], plots[pair.plot]);
    plot_taken_[pair.plot] = true;
    track_corrected_[pair.track] = true;
  }
  for (std::size_t t = 0; t < tracks_.size(); ++t)
    if (!track_corrected_[t]) ++tracks_[t].misses;
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](Track const& track) { return track.misses > config_.max_misses; }),
                tracks_.end());

  for (std::size_t p = 0; p < plots.size(); ++p) {
    if (plot_taken_[p]) continue;
    tracks_.push_back({next_id_++, 0, out_of(plots[p]), along_of(plots[p]), 0.0, 0.0, plots[p].time,
                       1, 0, config_.confirm_hits <= 1});
  }
}

bool Tracker::update_track(std::size_t id, std::vector<Detection> const& detections) {
  auto track = std::find_if(tracks_.begin(), tracks_.end(), [&](Track const& t) { return t.id == id; });
  if (track == tracks_.end()) return false;  // dropped since the pulse was scheduled
  std::optional<Plot> closest;
  double closest_distance = config_.track_gate;
  for (auto const& detection : detections) {
    Plot plot{detection.theta, detection.range, detection.strength, detection.time, 1};
    auto [out, along] = predict(*track, plot.time);
    double distance = std::hypot(out_of(plot) - out, along_of(plot) - along);
    if (distance <= closest_distance) {
      closest_distance = distance;
      closest = plot;
    }
  }
  if (!closest) return false;
  correct(*track, *closest);
  return true;
}

void Tracker::correct(Track& track, Plot const& plot) {
  double elapsed = plot.time - track.time;
  auto [out, along] = predict(track, plot.time);
  double error_out = out_of(plot) - out;
  double error_along = along_of(plot) - along;
  track.out = out + config_.track_alpha * error_out;
  track.along = along + config_.track_alpha * error_along;
  if (elapsed > 0.0) {
    track.v_out += config_.track_beta * error_out / elapsed;
    track.v_along += config_.track_beta * error_along / elapsed;
  }
  track.time = plot.time;
  ++track.hits;
  track.misses = 0;
  track.confirmed = track.hits >= config_.confirm_hits;
}
