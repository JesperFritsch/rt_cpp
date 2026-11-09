#pragma once
#include <rt/radar/types.hpp>
#include <cmath>
#include <utility>
#include <vector>

// alpha-beta tracker, updated once per scan with that scan's plots: each track is predicted to the
// plot's time, takes the closest plot within the gate and moves part of the way toward it;
// plots that no track takes start new tracks
class Tracker {
public:
  explicit Tracker(RadarConfig const& config);

  std::vector<Track> const& tracks() const { return tracks_; }

  void update(std::vector<Plot> const& plots);

  // corrects track `id` with the detection from a pulse aimed at it that is closest to its prediction,
  // if any is within the gate; returns whether the track was corrected
  // a pulse that finds nothing is no miss, misses are counted per scan by update
  bool update_track(std::size_t id, std::vector<Detection> const& detections);

private:
  struct Pair {
    double distance;  // m, from the track's prediction to the plot
    std::size_t track;
    std::size_t plot;
  };

  static double out_of(Plot const& plot) { return plot.range * std::cos(plot.theta); }
  static double along_of(Plot const& plot) { return plot.range * std::sin(plot.theta); }

  static std::pair<double, double> predict(Track const& track, double time) {
    double elapsed = time - track.time;
    return {track.out + track.v_out * elapsed, track.along + track.v_along * elapsed};
  }

  void correct(Track& track, Plot const& plot);

  RadarConfig config_;
  std::vector<Track> tracks_;
  std::vector<Pair> pairs_;
  std::vector<bool> plot_taken_;
  std::vector<bool> track_corrected_;
  std::size_t next_id_ = 1;
};
