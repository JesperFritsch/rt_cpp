#include <rt/radar/sim.hpp>
#include <rt/radar/waveform.hpp>
#include <algorithm>
#include <cmath>

std::vector<Cell> line(std::size_t x0, std::size_t y0, std::size_t x1, std::size_t y1) {
  std::vector<Cell> out;
  for (std::size_t x = x0; x <= x1; ++x)
    for (std::size_t y = y0; y <= y1; ++y)
      out.push_back({x, y});
  return out;
}

std::vector<Cell> disk(std::size_t center_x, std::size_t center_y, std::size_t radius) {
  std::vector<Cell> out;
  auto radius_squared = static_cast<double>(radius * radius);
  for (std::size_t x = center_x - radius; x <= center_x + radius; ++x)
    for (std::size_t y = center_y - radius; y <= center_y + radius; ++y) {
      double dx = static_cast<double>(x) - static_cast<double>(center_x);
      double dy = static_cast<double>(y) - static_cast<double>(center_y);
      if (dx * dx + dy * dy <= radius_squared) out.push_back({x, y});
    }
  return out;
}

SourceFn sine(double freq_hz, double amplitude) {
  return [=](double t) { return amplitude * std::sin(2.0 * pi * freq_hz * t); };
}

SourceFn gaussian_pulse(double t_center, double width, double amplitude) {
  return [=](double t) {
    double widths_from_center = (t - t_center) / width;
    return amplitude * std::exp(-widths_from_center * widths_from_center);
  };
}

SourceFn burst(double freq_hz, int cycles, double amplitude) {
  double duration = static_cast<double>(cycles) / freq_hz;
  return [=](double t) { return t < duration ? amplitude * std::sin(2.0 * pi * freq_hz * t) : 0.0; };
}

// ---------------------------------------------------------------------------
// PhasedArray

PhasedArray::PhasedArray(std::vector<Cell> cells, double cell_size, double length_cells, double theta)
  : state_(std::make_shared<State>()) {
  auto& state = *state_;
  state.cell_size = cell_size;
  state.length_cells = length_cells;
  state.theta = theta;
  state.cells = std::move(cells);
}

void PhasedArray::fire(PulseRequest p_req) {
  auto& state = *state_;
  state.pulses.push_back({p_req.fire_time, std::sin(p_req.theta), p_req.wave});
}

void PhasedArray::prune_pulses(double sim_time) {
  double max_delay = state_->length_cells * state_->cell_size / speed_of_light;
  state_->pulses.erase(
    std::remove_if(state_->pulses.begin(), state_->pulses.end(),
                   [&](Pulse const& pulse) {
                     return sim_time > pulse.fire_time + pulse.wave.duration + max_delay;
                   }),
    state_->pulses.end());
}

SourceFn PhasedArray::element_fn(double offset_cells) const {
  return [state = state_, offset_cells](double t) {
    float sum = 0.0f;
    for (auto const& pulse : state->pulses) {
      // distance in cells from the element that fires first, so every delay is >= 0:
      // the left end for positive theta, the right end for negative theta
      double cells_from_first_to_fire = pulse.sin_theta >= 0.0 ? offset_cells : offset_cells - state->length_cells;
      double delay = cells_from_first_to_fire * state->cell_size * pulse.sin_theta / speed_of_light;
      // how far this element is into its delayed copy of the pulse
      double time_into_pulse = t - pulse.fire_time - delay;
      if (time_into_pulse >= 0.0 && time_into_pulse <= pulse.wave.duration)
        sum += wave_contrib(time_into_pulse, pulse.wave);
    }
    return sum;
  };
}

// ---------------------------------------------------------------------------
// RadarSim

RadarSim::RadarSim(std::unique_ptr<FdtdBackend> backend, double cell_size, float courant) :
  backend_(std::move(backend)),
  cell_size_(cell_size),
  dt_(courant * cell_size / speed_of_light),
  width_(backend_->width()),
  height_(backend_->height()),
  cell_features(width_ * height_) // use the members to not get warning about unused height_
{}

void RadarSim::rebuild_layout() {
  std::vector<Cell> source_cells;
  for (auto const& source : sources)
    source_cells.insert(source_cells.end(), source.cells.begin(), source.cells.end());
  backend_->set_source_cells(std::move(source_cells));

  std::vector<Cell> probe_cells;
  array_probe_offsets_.clear();
  for (auto const& array : arrays) {
    array_probe_offsets_.push_back(probe_cells.size());
    probe_cells.insert(probe_cells.end(), array.cells().begin(), array.cells().end());
  }
  receiver_probe_offset_ = probe_cells.size();
  for (auto const& receiver : receivers) probe_cells.push_back(receiver.cell);
  backend_->set_probe_cells(std::move(probe_cells));
  layout_dirty_ = false;
}

void RadarSim::create_source(std::vector<Cell> cells, SourceFn fn) {
  for (auto const& cell : cells) cell_features[cell.y * width_ + cell.x] |= FEATURE_TX;
  sources.push_back({std::move(cells), std::move(fn)});
  layout_dirty_ = true;
}

PhasedArray RadarSim::create_phased_array(Cell first, std::size_t n_elements, std::size_t spacing, bool x_plane, double theta) {
  auto length_cells = static_cast<double>((n_elements - 1) * spacing);
  std::vector<Cell> cells;
  for (std::size_t i = 0; i < n_elements; ++i) {
    if (x_plane) {
      cells.push_back({first.x + i * spacing, first.y});
    } else {
      cells.push_back({first.x, first.y + i * spacing});
    }
  }
  PhasedArray array(cells, cell_size_, length_cells, theta);
  for (auto const& cell : cells) {
      auto cell_offset = x_plane ? cell.x - first.x : cell.y - first.y;
      create_source({cell}, array.element_fn(static_cast<double>(cell_offset)));
  }
  arrays.push_back(array);
  layout_dirty_ = true;
  return array;
}

std::size_t RadarSim::create_receiver(Cell cell) {
  receivers.push_back({cell, {}});
  cell_features[cell.y * width_ + cell.x] |= FEATURE_RX;
  layout_dirty_ = true;
  return receivers.size() - 1;
}

void RadarSim::add_metal(std::vector<Cell> const& cells) {
  for (auto const& cell : cells) cell_features[cell.y * width_ + cell.x] |= FEATURE_METAL;
  backend_->set_pec(cells, true);
}

void RadarSim::remove_metal(std::vector<Cell> const& cells) {
  for (auto const& cell : cells) cell_features[cell.y * width_ + cell.x] &= static_cast<uint8_t>(~FEATURE_METAL);
  backend_->set_pec(cells, false);
}

void RadarSim::step(std::size_t n) {
  if (layout_dirty_) rebuild_layout();
  std::size_t n_sources = backend_->source_count();
  std::size_t n_probes = backend_->probe_count();

  source_values_.resize(n * n_sources);
  for (std::size_t i = 0; i < n; ++i) {
    double now = static_cast<double>(step_ + i) * dt_;
    float* values = source_values_.data() + i * n_sources;
    for (auto& source : sources) {
      float value = source.fn(now);
      for (std::size_t c = 0; c < source.cells.size(); ++c) *values++ = value;
    }
  }

  probe_rows_.resize(n * n_probes);
  backend_->step(n, source_values_.data(), probe_rows_.data());
  step_ += n;
  batch_size_ = n;

  for (std::size_t i = 0; i < n; ++i) {
    float const* row = probe_rows_.data() + i * n_probes + receiver_probe_offset_;
    for (std::size_t r = 0; r < receivers.size(); ++r) receivers[r].samples.push_back(row[r]);
  }
  for (auto& array : arrays) array.prune_pulses(time());
}
