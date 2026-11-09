#pragma once
// the radar's computer: schedules pulses, processes each pulse's window into detections,
// merges them into plots per scan and keeps the tracks
// a template over the ring sizes and the element count, so it lives entirely in this header
#include <rt/radar/scheduler.hpp>
#include <rt/radar/tracker.hpp>
#include <rt/radar/types.hpp>
#include <rt/radar/waveform.hpp>
#include <rt/ring_buffer_spsc.hpp>
#include <rt/terminal_plot.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// runs on the worker thread: takes sample batches from the sim, answers with pulse requests
// and reports what it finds as detections per pulse and plots per scan
template<std::size_t SAMPLE_RING_SIZE, std::size_t REQUEST_RING_SIZE, std::size_t N_ELEMENTS,
         std::size_t DETECTION_RING_SIZE, std::size_t PLOT_RING_SIZE, std::size_t TRACK_RING_SIZE>
class RadarProcessor {
public:
  RadarProcessor(rt::SPSCRing<PulseRequest, REQUEST_RING_SIZE>& pulse_requests,
                 rt::SPSCRing<SampleBatch<N_ELEMENTS>, SAMPLE_RING_SIZE>& sample_batches,
                 rt::SPSCRing<Detection, DETECTION_RING_SIZE>& detections_out,
                 rt::SPSCRing<Plot, PLOT_RING_SIZE>& plots_out,
                 rt::SPSCRing<Track, TRACK_RING_SIZE>& tracks_out,
                 RadarConfig config,
                 Geometry geometry)
    : pulse_requests_(pulse_requests),
      sample_batches_(sample_batches),
      detections_out_(detections_out),
      plots_out_(plots_out),
      tracks_out_(tracks_out),
      tracker_(config),
      config_(config),
      geometry_(std::move(geometry)),
      dt_(geometry_.dt),
      pri_steps_(config_.pri_steps(geometry_)),
      scheduler_(pri_steps_, dt_, config_.search_waveform(), config_.track_waveform())
  {
    assert(geometry_.elements.size() == N_ELEMENTS);
    // pulses are requested one PRI ahead, so at most two fire steps are in flight
    in_flight_.reserve(8);
    requests_.reserve(4);
    window_.resize(pri_steps_ * N_ELEMENTS);
    beam_.resize(pri_steps_);
    auto const& first = geometry_.elements.front();
    for (auto const& element : geometry_.elements)
      element_offsets_.push_back(std::hypot(element.x - first.x, element.y - first.y));
    auto reference_len = static_cast<std::size_t>(std::ceil(config_.pulse_duration / dt_));
    reference_.reserve(reference_len);
    reference_quadrature_.reserve(reference_len);
    compressed_.resize(pri_steps_);
    plot_envelope_.resize(pri_steps_);
    plot_received_.resize(pri_steps_);
    detections_.reserve(16);
    peak_lags_.reserve(16);
    angle_response_.reserve(128);
    scan_detections_.reserve(64);
    merged_.reserve(64);
    plots_.reserve(16);
    scheduler_.set_angles(config_.max_scan_angle_r, config_.max_scan_angle_r / 10.0);
    scheduler_.set_track_interval(config_.track_interval);
    scheduler_.set_simultaneous_tracks(config_.simultaneous_tracks);
  }

  std::size_t pri_steps() const { return pri_steps_; }

  // steps for one sweep over all scan angles, the longest it takes when Track pulses are interleaved
  std::size_t scan_steps() const { return pri_steps_ * scheduler_.pulses_per_scan(); }

  void run() {
    while ( auto batch = sample_batches_.pop_blocking() ) handle(*batch);
  }

  void handle(SampleBatch<N_ELEMENTS> const& batch) {
    requests_.clear();
    scheduler_.on_step(batch.step, requests_);
    for (auto const& request : requests_) {
      while (!pulse_requests_.push(request)) {}
      in_flight_.push_back(request);
    }
    collect(batch);
  }

private:
  std::size_t fire_step(PulseRequest const& req) const {
    return static_cast<std::size_t>(std::llround(req.fire_time / dt_));
  }

  // copies the batch into the window of the oldest in-flight pulses;
  // a window is the pri_steps steps from its fire step, so windows lie back to back,
  // and every pulse fired at that step shares it
  void collect(SampleBatch<N_ELEMENTS> const& batch) {
    if (in_flight_.empty()) return;
    std::size_t first_step = fire_step(in_flight_.front());
    if (batch.step < first_step) return;  // before the oldest pulses fired

    std::size_t row = batch.step - first_step;
    std::copy(batch.samples.begin(), batch.samples.end(), window_.begin() + static_cast<std::ptrdiff_t>(row * N_ELEMENTS));
    if (row + 1 < pri_steps_) return;

    // in_flight_ is in fire step order, so the pulses sharing this window come first
    auto shared_end = std::find_if(in_flight_.begin(), in_flight_.end(),
                                   [&](PulseRequest const& pulse) { return fire_step(pulse) != first_step; });
    // each pulse's own transmission is in the window, so every one is blind as long as the most steered
    double window_steer_time = 0.0;
    for (auto pulse = in_flight_.begin(); pulse != shared_end; ++pulse)
      window_steer_time = std::max(window_steer_time, steer_time(pulse->theta));
    for (auto pulse = in_flight_.begin(); pulse != shared_end; ++pulse)
      on_window(*pulse, window_steer_time);
    in_flight_.erase(in_flight_.begin(), shared_end);
  }

  // samples `wave` at the sim rate in phase and in quadrature, what the matched filter looks for;
  // rebuilt for every pulse, which costs little next to the matched filter itself
  void build_reference(Waveform const& wave) {
    reference_wave_ = wave;
    reference_.clear();
    reference_quadrature_.clear();
    auto reference_len = static_cast<std::size_t>(std::ceil(wave.duration / dt_));
    for (std::size_t i = 0; i < reference_len; ++i) {
      reference_.push_back(wave_contrib(static_cast<double>(i) * dt_, wave));
      reference_quadrature_.push_back(wave_contrib(static_cast<double>(i) * dt_, wave, pi / 2.0));
    }
  }

  // time from the first to the last element firing a pulse steered at theta
  double steer_time(double theta) const {
    return geometry_.array_length * std::abs(std::sin(theta)) / speed_of_light;
  }

  // delay and sum toward sin_theta: beam_[row] is what the array center receives at row,
  // so echoes from that direction add up in phase and others partly cancel
  // only rows in [first_row, end_row) are formed, the rest of beam_ is left as it was
  // returns the largest |shift| applied to an element, in steps
  std::size_t beamform(double sin_theta, std::size_t first_row = 0, std::size_t end_row = SIZE_MAX) {
    end_row = std::min(end_row, pri_steps_);
    std::fill(beam_.begin() + static_cast<std::ptrdiff_t>(first_row), beam_.begin() + static_cast<std::ptrdiff_t>(end_row), 0.0f);
    double center = geometry_.array_length / 2.0;
    auto rows = static_cast<std::ptrdiff_t>(pri_steps_);
    std::size_t max_shift = 0;
    for (std::size_t element = 0; element < N_ELEMENTS; ++element) {
      // an echo from theta reaches elements further along the array earlier than the center
      double lead = (element_offsets_[element] - center) * sin_theta / speed_of_light;
      auto shift = static_cast<std::ptrdiff_t>(std::llround(-lead / dt_));
      max_shift = std::max(max_shift, static_cast<std::size_t>(std::abs(shift)));
      for (auto row = static_cast<std::ptrdiff_t>(first_row); row < static_cast<std::ptrdiff_t>(end_row); ++row) {
        std::ptrdiff_t source_row = row + shift;
        if (source_row < 0 || source_row >= rows) continue;
        beam_[static_cast<std::size_t>(row)] += window_[static_cast<std::size_t>(source_row) * N_ELEMENTS + element];
      }
    }
    return max_shift;
  }

  // correlates the beam with the reference pulse and its quadrature copy; the magnitude of the two
  // is the envelope, large when an echo reaches the array center `lag` steps after the fire step,
  // without the carrier swing a single correlation has
  float matched_filter(std::size_t lag) const {
    float in_phase = 0.0f;
    float quadrature = 0.0f;
    for (std::size_t k = 0; k < reference_.size(); ++k) {
      in_phase += beam_[lag + k] * reference_[k];
      quadrature += beam_[lag + k] * reference_quadrature_[k];
    }
    return std::hypot(in_phase, quadrature);
  }

  // receive-beam scan: beamforms toward trial angles around beam_theta and returns the one where
  // the echo at `lag` is strongest; the transmit beam is the same for every trial angle, so it
  // scales the response but does not move its peak
  // overwrites the rows of beam_ the matched filter reads at `lag`
  double estimate_theta(double beam_theta, std::size_t lag) {
    // scan about one beam width each way; the beam widens as 1 / cos when steered off broadside
    double center_freq = (reference_wave_.f_start + reference_wave_.f_end) / 2.0;
    double beam_width = speed_of_light / center_freq / (geometry_.array_length * std::cos(beam_theta));
    auto half_steps = static_cast<std::ptrdiff_t>(std::ceil(beam_width / config_.angle_step_r));

    angle_response_.clear();
    std::size_t best = 0;
    for (std::ptrdiff_t i = -half_steps; i <= half_steps; ++i) {
      double trial = beam_theta + static_cast<double>(i) * config_.angle_step_r;
      float response = 0.0f;
      if (std::abs(trial) < pi / 2.0) {
        beamform(std::sin(trial), lag, lag + reference_.size());
        response = matched_filter(lag);
      }
      angle_response_.push_back(response);
      if (response > angle_response_[best]) best = angle_response_.size() - 1;
    }

    // a parabola through the best trial angle and its neighbours puts the peak between steps
    double offset = 0.0;
    if (best > 0 && best + 1 < angle_response_.size()) {
      double before = angle_response_[best - 1];
      double at = angle_response_[best];
      double after = angle_response_[best + 1];
      double curvature = before - 2.0 * at + after;
      if (curvature < 0.0) offset = 0.5 * (before - after) / curvature;
    }
    double best_step = static_cast<double>(best) - static_cast<double>(half_steps) + offset;
    return beam_theta + best_step * config_.angle_step_r;
  }

  // window_ holds pri_steps rows of N_ELEMENTS samples, row 0 is the fire step
  // fills detections_ with every echo past the blind range that stands out from the rest of the envelope
  // window_steer_time is the largest steer_time of the pulses sharing the window
  void on_window(PulseRequest const& pulse, double window_steer_time) {
    // the search pulses wrap from the largest angle back to the smallest, which ends a scan;
    // Track pulses are aimed anywhere, so only search pulses count
    bool search = pulse.mode == PulseMode::Search;
    if (search && pulse.theta < last_beam_theta_) finish_scan();
    if (search) last_beam_theta_ = pulse.theta;

    // look for this pulse's own waveform, so echoes of other pulses in the window barely register
    build_reference(pulse.wave);
    double sin_theta = std::sin(pulse.theta);
    std::size_t max_shift = beamform(sin_theta);

    // the steering delays make the array center transmit this many steps after the fire step
    double tx_offset_rows = steer_time(pulse.theta) / 2.0 / dt_;
    // our own pulses are on some element until the last one has fired them in full,
    // and the receive shifts can move that up to max_shift rows later in the beam
    // it also takes up to array_length / c to reach the far elements through the array
    double crossing_time = geometry_.array_length / speed_of_light;
    auto blind_rows = static_cast<std::size_t>(std::ceil((config_.pulse_duration + window_steer_time + crossing_time) / dt_)) + max_shift;

    std::size_t final_lag = pri_steps_ - reference_.size();
    if (blind_rows > final_lag) return;
    double compressed_sum = 0.0;
    float compressed_max = 0.0f;
    for (std::size_t lag = blind_rows; lag <= final_lag; ++lag) {
      compressed_[lag] = matched_filter(lag);
      compressed_sum += static_cast<double>(compressed_[lag]);
      compressed_max = std::max(compressed_max, compressed_[lag]);
    }
    double mean = compressed_sum / static_cast<double>(final_lag - blind_rows + 1);
    auto threshold = static_cast<float>(config_.detection_factor * mean);

    // each run of lags above the threshold is one echo, detected at its strongest lag
    detections_.clear();
    peak_lags_.clear();
    std::size_t peak_lag = 0;
    float peak = 0.0f;
    for (std::size_t lag = blind_rows; lag <= final_lag + 1; ++lag) {
      bool above = lag <= final_lag && compressed_[lag] > threshold;
      if (above && compressed_[lag] > peak) {
        peak = compressed_[lag];
        peak_lag = lag;
      }
      if (!above && peak > 0.0f) {
        // range from the array center: time from its transmit to the echo at it, there and back
        double range = speed_of_light * (static_cast<double>(peak_lag) - tx_offset_rows) * dt_ / 2.0;
        // theta is the beam direction until estimate_theta below refines it
        detections_.push_back({pulse.theta, range, static_cast<float>(static_cast<double>(peak) / mean),
                               pulse.fire_time, pulse.theta});
        peak_lags_.push_back(peak_lag);
        peak = 0.0f;
      }
    }
    // before estimate_theta, which overwrites parts of beam_; one search pulse per PRI, so one plot per PRI
    if (search) draw_plots(pulse, blind_rows, final_lag, threshold, mean, tx_offset_rows);

    // after the search above, since estimate_theta overwrites parts of beam_
    for (std::size_t i = 0; i < detections_.size(); ++i)
      detections_[i].theta = estimate_theta(pulse.theta, peak_lags_[i]);

    if (search)
      std::printf("pulse at step %zu, theta %+6.1f deg:", fire_step(pulse), pulse.theta * 180.0 / pi);
    else
      std::printf("pulse at step %zu, theta %+6.1f deg, track #%zu:", fire_step(pulse), pulse.theta * 180.0 / pi,
                  pulse.track_id);
    // with nothing detected, the strongest response shows how far below the threshold it is
    if (detections_.empty()) std::printf(" -  (max x%.1f)", static_cast<double>(compressed_max) / mean);
    for (auto const& detection : detections_)
      std::printf("  %.2f m at %+.1f deg (x%.1f)", detection.range, detection.theta * 180.0 / pi,
                  static_cast<double>(detection.strength));
    std::printf("\n");

    // detections are only drawn, so drop them rather than stall processing if the sim falls behind
    for (auto const& detection : detections_) (void)detections_out_.push(detection);

    // search detections wait for the scan to end; a Track pulse corrects its track right away,
    // which is what makes a track update more often than once per scan
    if (search) {
      scan_detections_.insert(scan_detections_.end(), detections_.begin(), detections_.end());
    } else if (tracker_.update_track(pulse.track_id, detections_)) {
      scheduler_.set_tracks(tracker_.tracks());
      send_tracks();
    }
  }

  // amplitude `value` in dB relative to `reference`, no lower than -config_.plot_db_range
  float to_db(float value, float reference) const {
    double floor = std::pow(10.0, -config_.plot_db_range / 20.0);
    double ratio = std::max(static_cast<double>(std::abs(value)) / static_cast<double>(reference), floor);
    return static_cast<float>(20.0 * std::log10(ratio));
  }

  // redraws the top of the terminal over the whole window: the middle element's raw samples, then
  // the pulse-compressed envelope with the threshold dotted, a ^ under each detection and a | where
  // the blind zone ends
  // both are scaled to their peak after the blind zone, so our own transmission at the start is
  // clipped instead of squeezing every echo flat; config_.plot_db shows them in dB instead
  void draw_plots(PulseRequest const& pulse, std::size_t blind_rows, std::size_t final_lag,
                  float threshold, double mean, double tx_offset_rows) {
    std::size_t cols = dashboard_.width();
    if (cols < 20) return;
    bool db = config_.plot_db;
    auto db_range = static_cast<float>(config_.plot_db_range);

    std::vector<std::string> lines;
    lines.reserve(plot_lines);
    // a title wider than the terminal would wrap into the plot below it
    auto title = [&](std::string text) { return text.substr(0, cols); };

    // one element, not the beam, so echoes from every direction show; window_ is time-major
    constexpr std::size_t element = N_ELEMENTS / 2;
    float peak_received = 0.0f;
    for (std::size_t row = 0; row < pri_steps_; ++row) {
      plot_received_[row] = window_[row * N_ELEMENTS + element];
      if (row >= blind_rows) peak_received = std::max(peak_received, std::abs(plot_received_[row]));
    }
    if (peak_received == 0.0f) peak_received = 1.0f;
    BrailleCanvas received(cols, plot_height);
    if (db) {
      for (float& value : plot_received_) value = to_db(value, peak_received);
      plot_series(received, plot_received_.data(), plot_received_.size(), -db_range, 0.0f);
    } else {
      plot_series(received, plot_received_.data(), plot_received_.size(), -peak_received, peak_received);
    }
    lines.push_back(title(format_text("received at element %zu, pulse at step %zu toward %+.1f deg, %s %.3g after the blind zone",
                                element, fire_step(pulse), pulse.theta * 180.0 / pi,
                                db ? "0 dB =" : "scaled to", static_cast<double>(peak_received))));
    for (std::size_t r = 0; r < received.rows(); ++r) lines.push_back(received.row(r));

    // the envelope is only computed between the blind rows and the last lag, zero elsewhere
    std::fill(plot_envelope_.begin(), plot_envelope_.end(), 0.0f);
    std::copy(compressed_.begin() + static_cast<std::ptrdiff_t>(blind_rows),
              compressed_.begin() + static_cast<std::ptrdiff_t>(final_lag + 1),
              plot_envelope_.begin() + static_cast<std::ptrdiff_t>(blind_rows));
    float peak_envelope = *std::max_element(plot_envelope_.begin(), plot_envelope_.end());
    if (peak_envelope == 0.0f) peak_envelope = 1.0f;
    BrailleCanvas compressed(cols, plot_height);
    if (db) {
      for (float& value : plot_envelope_) value = to_db(value, peak_envelope);
      float threshold_db = to_db(threshold, peak_envelope);
      float top = std::max(0.0f, threshold_db) + 3.0f;
      plot_series(compressed, plot_envelope_.data(), plot_envelope_.size(), -db_range, top);
      plot_level(compressed, threshold_db, -db_range, top);
    } else {
      float top = 1.1f * std::max(peak_envelope, threshold);
      plot_series(compressed, plot_envelope_.data(), plot_envelope_.size(), 0.0f, top);
      plot_level(compressed, threshold, 0.0f, top);
    }
    lines.push_back(title(format_text("pulse compressed (beam toward %+.1f deg), threshold x%.1f dotted, peak x%.1f%s, %zu detections",
                                pulse.theta * 180.0 / pi, config_.detection_factor,
                                static_cast<double>(peak_envelope) / mean, db ? " = 0 dB" : "", detections_.size())));
    for (std::size_t r = 0; r < compressed.rows(); ++r) lines.push_back(compressed.row(r));

    // range axis: the range at each end of the window, | where the blind zone ends, ^ under each detection
    auto range_at = [&](double lag) { return speed_of_light * (lag - tx_offset_rows) * dt_ / 2.0; };
    std::string axis(cols, ' ');
    axis[std::min(cols - 1, blind_rows * cols / pri_steps_)] = '|';
    for (std::size_t lag : peak_lags_) axis[std::min(cols - 1, lag * cols / pri_steps_)] = '^';
    std::string left = format_text("%.1f m", range_at(0.0));
    std::string right = format_text("%.1f m", range_at(static_cast<double>(pri_steps_ - 1)));
    if (left.size() + right.size() + 2 < cols) {
      axis.replace(0, left.size(), left);
      axis.replace(cols - right.size(), right.size(), right);
    }
    lines.push_back(axis);

    dashboard_.draw(lines);
  }

  // sends every track to the sim under a new report number, so it replaces the previous set
  void send_tracks() {
    ++report_;
    for (auto track : tracker_.tracks()) {
      track.report = report_;
      (void)tracks_out_.push(track);
    }
  }

  // merges the scan's detections into plots: the strongest detection not yet merged starts a plot
  // and takes every unmerged detection within merge_distance of it, so a target that several
  // pulses saw becomes one plot
  void finish_scan() {
    std::sort(scan_detections_.begin(), scan_detections_.end(),
              [](Detection const& a, Detection const& b) { return a.strength > b.strength; });
    merged_.assign(scan_detections_.size(), false);
    plots_.clear();

    // positions relative to the array center: out along broadside, along the array
    auto out_of = [](Detection const& detection) { return detection.range * std::cos(detection.theta); };
    auto along_of = [](Detection const& detection) { return detection.range * std::sin(detection.theta); };

    for (std::size_t seed = 0; seed < scan_detections_.size(); ++seed) {
      if (merged_[seed]) continue;
      double seed_out = out_of(scan_detections_[seed]);
      double seed_along = along_of(scan_detections_[seed]);
      double weight_sum = 0.0;
      double out_sum = 0.0;
      double along_sum = 0.0;
      double latest = 0.0;
      std::size_t count = 0;
      for (std::size_t i = seed; i < scan_detections_.size(); ++i) {
        if (merged_[i]) continue;
        auto const& detection = scan_detections_[i];
        double out = out_of(detection);
        double along = along_of(detection);
        if (std::hypot(out - seed_out, along - seed_along) > config_.merge_distance) continue;
        merged_[i] = true;
        auto weight = static_cast<double>(detection.strength);
        weight_sum += weight;
        out_sum += weight * out;
        along_sum += weight * along;
        latest = std::max(latest, detection.time);
        ++count;
      }
      double out = out_sum / weight_sum;
      double along = along_sum / weight_sum;
      plots_.push_back({std::atan2(along, out), std::hypot(out, along), scan_detections_[seed].strength, latest, count});
    }

    std::printf("scan done:");
    if (plots_.empty()) std::printf(" -");
    for (auto const& plot : plots_)
      std::printf("  %.2f m at %+.1f deg (%zu detections)", plot.range, plot.theta * 180.0 / pi, plot.detections);
    std::printf("\n");

    for (auto const& plot : plots_) (void)plots_out_.push(plot);
    scan_detections_.clear();

    tracker_.update(plots_);
    scheduler_.set_tracks(tracker_.tracks());
    std::printf("tracks:");
    if (tracker_.tracks().empty()) std::printf(" -");
    for (auto const& track : tracker_.tracks())
      std::printf("  #%zu %.2f m %+.1f deg%s", track.id, std::hypot(track.out, track.along),
                  std::atan2(track.along, track.out) * 180.0 / pi, track.confirmed ? "" : " (tentative)");
    std::printf("\n");
    send_tracks();
  }

  rt::SPSCRing<PulseRequest, REQUEST_RING_SIZE>& pulse_requests_;           // to the sim
  rt::SPSCRing<SampleBatch<N_ELEMENTS>, SAMPLE_RING_SIZE>& sample_batches_; // from the sim
  rt::SPSCRing<Detection, DETECTION_RING_SIZE>& detections_out_;            // to the sim, for drawing
  rt::SPSCRing<Plot, PLOT_RING_SIZE>& plots_out_;                           // to the sim, for drawing
  rt::SPSCRing<Track, TRACK_RING_SIZE>& tracks_out_;                        // to the sim, for drawing
  Tracker tracker_;
  std::size_t report_ = 0;                  // sets of tracks sent to the sim so far
  std::vector<Detection> scan_detections_;  // every detection since the scan started
  std::vector<bool> merged_;                // finish_scan's bookkeeping, same order as scan_detections_
  std::vector<Plot> plots_;                 // from the latest scan
  double last_beam_theta_ = -pi;            // lower than any scan angle, so the first pulse starts no scan
  std::vector<PulseRequest> in_flight_;  // requested pulses whose window is not complete, oldest first
  std::vector<PulseRequest> requests_;   // handle's scratch for the scheduler's requests
  std::vector<float> window_;            // pri_steps x N_ELEMENTS, time-major
  Waveform reference_wave_{};            // the waveform reference_ was built from
  std::vector<float> reference_;         // transmitted pulse, one sample per step
  std::vector<float> reference_quadrature_;  // reference_ with the carrier shifted 90 degrees
  std::vector<float> compressed_;          // matched filter magnitude per lag, pri_steps long
  static constexpr std::size_t plot_height = 10;                  // lines per plot
  static constexpr std::size_t plot_lines = 2 * (1 + plot_height) + 1;  // two titled plots and the range axis
  TerminalDashboard dashboard_{plot_lines};
  std::vector<float> plot_envelope_;     // envelope_ with the lags it was not computed for zeroed, pri_steps long
  std::vector<float> plot_received_;     // one element's samples from window_, pri_steps long
  std::vector<Detection> detections_;    // from the latest window
  std::vector<std::size_t> peak_lags_;   // envelope lag of each detection, same order
  std::vector<float> angle_response_;    // estimate_theta's matched filter output per trial angle
  std::vector<float> beam_;              // window_ beamformed toward the pulse's theta, pri_steps long
  std::vector<double> element_offsets_;  // m from the first element along the array
  RadarConfig config_;
  Geometry geometry_;
  double dt_;
  std::size_t pri_steps_;
  PulseScheduler scheduler_;
};
