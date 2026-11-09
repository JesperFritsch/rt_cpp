#pragma once
// the data the radar's parts pass between them, and the configuration they share
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double speed_of_light = 299'792'458.0;  // m/s

struct Waveform {
  double f_start;
  double f_end;
  double duration;
  double amplitude;
  double taper;  // share of the duration at each end that ramps in or out, 0 is a rectangular pulse
};

enum class PulseMode : uint8_t {
  Search,  // part of the scan over all angles
  Track,   // aimed at where a track is expected to be
};

struct PulseRequest {
  std::size_t step;  // step of the sample batch this request answers
  Waveform wave;
  double theta;
  double fire_time;
  PulseMode mode;
  std::size_t track_id;  // the track a Track pulse is aimed at, unused for Search
};

// ez at every array element for one simulation step
template<std::size_t N_ELEMENTS>
struct SampleBatch {
  std::size_t step;  // simulation step the samples were taken at
  std::array<float, N_ELEMENTS> samples;
};

// an echo found in one pulse's window
struct Detection {
  double theta;       // radians, direction the echo came from
  double range;       // m from the array center
  float strength;     // envelope peak over the window's mean envelope
  double time;        // s, fire time of the pulse
  double beam_theta;  // radians, direction the pulse was steered
};

// the detections from one scan that belong to the same echo, merged
struct Plot {
  double theta;            // radians, strength-weighted mean of the merged detections
  double range;            // m from the array center, same weighting
  float strength;          // of the strongest merged detection
  double time;             // s, fire time of the latest merged detection
  std::size_t detections;  // how many were merged
};

// a target followed over several scans; positions are relative to the array center,
// `out` along broadside and `along` toward the last element
struct Track {
  std::size_t id;
  std::size_t report;      // set when sent to the sim; all tracks sent together share it
  double out, along;       // m
  double v_out, v_along;   // m per s of sim time
  double time;             // s, time of the plot the track was last corrected with
  std::size_t hits;        // plots the track has taken
  std::size_t misses;      // scans in a row without a plot
  bool confirmed;          // has taken enough plots to be trusted
};

struct Position { double x, y; };  // meters

// what the processor knows about the sim, all in meters and seconds
struct Geometry {
  double cell_size;                // m per cell
  double dt;                       // s per step
  double domain_width;             // m
  double domain_height;            // m
  double array_length;             // m, first to last element
  std::vector<Position> elements;  // same order as SampleBatch::samples
};

struct RadarConfig {
  // chosen
  // search and track pulses chirp over separate bands, so when they share a window the matched
  // filter for one barely responds to the other's echoes
  double search_f_start = 1.2e9;
  double search_f_end = 1.45e9;
  double track_f_start = 1.55e9;
  double track_f_end = 1.8e9;
  double pulse_duration = 10e-9;
  double amplitude = 2.0;
  double pulse_taper = 0.5;                   // see Waveform::taper; a rectangular pulse spreads into the
                                               // other band and shows up there as ghosts
  double max_scan_angle_r = 35.0 * pi / 180.0;  // radians from broadside
  double pri_margin = 1.0;                     // < 1 trims the conservative listen time
  double detection_factor = 8.0;               // detect where the envelope exceeds this times its mean
  double angle_step_r = 0.5 * pi / 180.0;      // trial angle step when estimating a detection's direction
  double merge_distance = 0.5;                 // m, detections in one scan closer than this are one plot
  double track_gate = 1.0;                     // m, a track only takes plots this close to its prediction
  double track_alpha = 0.5;                    // share of the position error a track corrects per plot
  double track_beta = 0.3;                     // share of the error, per time since the last plot, added to velocity
  std::size_t confirm_hits = 1;                // plots before a track is confirmed
  std::size_t max_misses = 1;                  // scans without a plot before a track is dropped
  std::size_t track_interval = 2;              // with confirmed tracks, every track_interval-th pulse is a
                                               // Track pulse; 2 alternates, 0 turns track pulses off
  bool simultaneous_tracks = true;             // fire a Track pulse together with every Search pulse instead,
                                               // so tracking does not slow the scan; track_interval is unused
  bool plot_db = true;                         // terminal plots in dB relative to their peak instead of linear
  double plot_db_range = 60.0;                 // dB below the peak the plots reach down to

  // derived
  double blind_range() const        { return speed_of_light * pulse_duration / 2.0; }
  double pulse_length_m() const     { return speed_of_light * pulse_duration; }
  Waveform search_waveform() const  { return {search_f_start, search_f_end, pulse_duration, amplitude, pulse_taper}; }
  Waveform track_waveform() const   { return {track_f_start, track_f_end, pulse_duration, amplitude, pulse_taper}; }

  // the coarser of the two bands'
  double range_resolution() const {
    double bandwidth = std::min(std::abs(search_f_end - search_f_start), std::abs(track_f_end - track_f_start));
    return speed_of_light / (2.0 * bandwidth);
  }

  // upper bound on the time from fire_time until the last echo has arrived:
  // out and back are each at most one domain diagonal, steering delays the
  // last element by up to array_length * sin(max_scan_angle_r), plus the pulse itself
  double listen_time(Geometry const& geometry) const {
    double diagonal = std::hypot(geometry.domain_width, geometry.domain_height);
    double steer_extra = geometry.array_length * std::sin(max_scan_angle_r);
    return (2.0 * diagonal + steer_extra) / speed_of_light + pulse_duration;
  }

  std::size_t pri_steps(Geometry const& geometry) const {
    return static_cast<std::size_t>(std::ceil(listen_time(geometry) * pri_margin / geometry.dt));
  }
};
