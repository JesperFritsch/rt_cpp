#include <rt/cpu_backend.hpp>
#include <rt/gpu_backend.hpp>
#include <rt/radar/processor.hpp>
#include <rt/radar/sim.hpp>
#include <rt/ring_buffer_spsc.hpp>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <raylib.h>
#include <rlgl.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>

constexpr double visual_gain = 5.0;

// colors ez: red for positive, blue for negative, gray for metal
// ez is read from a shader storage buffer, so a backend that keeps it on the GPU needs no upload;
// the quad is drawn with a 1x1 texture only to get fragTexCoord from 0 to 1 across the window
constexpr const char* field_shader = R"(
#version 430
in vec2 fragTexCoord;
out vec4 finalColor;
layout(std430, binding = 0) readonly buffer Ez { float ez[]; };  // grid_size.x * grid_size.y, row-major
uniform ivec2 grid_size;
uniform sampler2D features_tex; // FEATURE_* bits: 0x1 = metal, 0x2 = receiver, 0x4 = transmitter
uniform float gain;
void main() {
  int features = int(texture(features_tex, fragTexCoord).r * 255 + 0.5);
  if ((features & 0x1) != 0) {
    finalColor = vec4(0.5, 0.5, 0.5, 1.0);
    return;
  }
  if ((features & 0x2) != 0) {
    finalColor = vec4(0.9, 0.95, 0.0, 1.0);
    return;
  }
  if ((features & 0x4) != 0) {
    finalColor = vec4(0.02, 0.87, 0.98, 1.0);
    return;
  }
  ivec2 cell = min(ivec2(fragTexCoord * vec2(grid_size)), grid_size - 1);
  float field = clamp(ez[cell.y * grid_size.x + cell.x] * gain, -1.0, 1.0);
  finalColor = vec4(max(field, 0.0), 0.0, max(-field, 0.0), 1.0);
}
)";

// prints a bar from a center line: left for negative, right for positive
// value is scaled so |value| == max_value fills half_width characters
void print_bar(float value, float max_value, int half_width) {
  float fraction = std::clamp(value / max_value, -1.0f, 1.0f);
  auto bar_length = static_cast<std::size_t>(std::lround(std::abs(fraction) * static_cast<float>(half_width)));
  auto side_width = static_cast<std::size_t>(half_width);

  std::string left(side_width, ' ');
  std::string right(side_width, ' ');
  if (fraction < 0)
    std::fill(left.end() - static_cast<std::ptrdiff_t>(bar_length), left.end(), '#');
  else
    std::fill(right.begin(), right.begin() + static_cast<std::ptrdiff_t>(bar_length), '#');

  std::printf("\r\033[2K%8.4f %s|%s", static_cast<double>(value), left.c_str(), right.c_str());
  std::fflush(stdout);
}


int main(int argc, char** argv) {
  // the field runs on the GPU unless started with --cpu
  bool use_cpu = argc > 1 && std::string(argv[1]) == "--cpu";

  // λ = 20 cells at 1.5 GHz with 1 cm cells, so the grid is 50 x 35 wavelengths
  constexpr std::size_t grid_width = 1000;
  constexpr std::size_t grid_height = 700;
  constexpr std::size_t border = 60;  // sponge, 3λ thick
  // window has the grid's aspect ratio so cells stay square
  constexpr int window_width = 1400;
  constexpr int window_height = static_cast<int>(window_width * grid_height / grid_width);
  InitWindow(window_width, window_height, "radar sim");
  SetTargetFPS(60);
  constexpr std::size_t steps_per_frame = 25;
  constexpr double cell_size = 0.01;   // 1 cm per cell
  constexpr std::size_t sample_ring_size = 128;
  constexpr std::size_t request_ring_size = 128;
  constexpr std::size_t detection_ring_size = 128;
  constexpr std::size_t plot_ring_size = 32;
  constexpr std::size_t track_ring_size = 64;

  constexpr float courant = 0.5f;
  constexpr float damp = 1.0f;

  // the GPU backend needs the OpenGL context, so it is built after InitWindow
  std::unique_ptr<FdtdBackend> backend;
  if (use_cpu) backend = std::make_unique<CpuBackend>(grid_width, grid_height, border, courant, damp);
  else         backend = std::make_unique<GpuBackend>(grid_width, grid_height, border, courant, damp);
  auto radar_sim = RadarSim(std::move(backend), cell_size, courant);
  std::printf("field on the %s\n", use_cpu ? "CPU" : "GPU");
  // a backend with ez on the CPU is uploaded into this buffer every frame
  constexpr auto ez_bytes = static_cast<unsigned>(grid_width * grid_height * sizeof(float));
  unsigned ez_upload_buffer = rlLoadShaderBuffer(ez_bytes, nullptr, RL_DYNAMIC_DRAW);
  Image quad_img = GenImageColor(1, 1, WHITE);
  Texture2D quad_tex = LoadTextureFromImage(quad_img);
  UnloadImage(quad_img);

  // vertical array of 16 elements at λ/2 spacing, 2λ in from the sponge, facing +x
  constexpr std::size_t n_elements = 16;
  constexpr std::size_t spacing = 10;
  constexpr std::size_t array_start_x = border + 40;
  constexpr std::size_t array_start_y = grid_height / 2 - (n_elements - 1) * spacing / 2;
  // ground plane λ/4 behind the elements, extending 1λ past each end of the array
  constexpr std::size_t plane_distance = 5;
  constexpr std::size_t plane_margin = 20;
  radar_sim.add_metal(line(array_start_x - plane_distance - 1, array_start_y - plane_margin,
                           array_start_x - plane_distance, array_start_y + (n_elements - 1) * spacing + plane_margin));

  // targets, drag them with the left mouse button
  struct Target { std::size_t x, y, radius; };  // cells
  std::vector<Target> targets{
    {grid_width * 3 / 4, grid_height * 3 / 4, 30},
    {grid_width * 6 / 7, grid_height * 1 / 4, 30},
  };
  for (auto const& target : targets) radar_sim.add_metal(disk(target.x, target.y, target.radius));
  std::optional<std::size_t> dragged;  // index into targets

  auto array = radar_sim.create_phased_array(
    {array_start_x, array_start_y },
    n_elements, 
    spacing,
    false
  );

  rt::SPSCRing<PulseRequest, request_ring_size> pulse_requests;
  rt::SPSCRing<SampleBatch<n_elements>, sample_ring_size> sample_batches;
  rt::SPSCRing<Detection, detection_ring_size> detections;
  rt::SPSCRing<Plot, plot_ring_size> plots;
  rt::SPSCRing<Track, track_ring_size> tracks;


  // features are uploaded here and again whenever a target is moved
  auto const& features = radar_sim.get_cell_features();
  std::vector<unsigned char> feature_bytes(features.begin(), features.end());
  Image features_img = {
    feature_bytes.data(),
    grid_width,
    grid_height,
    1,
    PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
  };
  Texture2D features_tex = LoadTextureFromImage(features_img);

  Shader shader = LoadShaderFromMemory(nullptr, field_shader);
  int features_loc = GetShaderLocation(shader, "features_tex");
  int gain_loc = GetShaderLocation(shader, "gain");
  float gain = static_cast<float>(visual_gain);
  SetShaderValue(shader, gain_loc, &gain, SHADER_UNIFORM_FLOAT);
  int grid_size[2] = {static_cast<int>(grid_width), static_cast<int>(grid_height)};
  SetShaderValue(shader, GetShaderLocation(shader, "grid_size"), grid_size, SHADER_UNIFORM_IVEC2);

  Geometry geometry{
    cell_size,
    radar_sim.dt(),
    static_cast<double>(grid_width) * cell_size,
    static_cast<double>(grid_height) * cell_size,
    static_cast<double>((n_elements - 1) * spacing) * cell_size,
    {}
  };
  for (auto const& cell : array.cells())
    geometry.elements.push_back({static_cast<double>(cell.x) * cell_size, static_cast<double>(cell.y) * cell_size});

  RadarConfig config;
  RadarProcessor processor(pulse_requests, sample_batches, detections, plots, tracks, config, geometry);
  std::printf("PRI %zu steps (%.1f ns), blind range %.2f m, range resolution %.2f m\n",
              processor.pri_steps(), static_cast<double>(processor.pri_steps()) * radar_sim.dt() * 1e9,
              config.blind_range(), config.range_resolution());
  auto worker_thread = std::thread([&]() { processor.run(); });

  // detections are kept for one scan and fade out over it
  double detection_lifetime = static_cast<double>(processor.scan_steps()) * radar_sim.dt();
  std::vector<Detection> shown_detections;
  // plots come once per scan, so the same lifetime shows each until the next one replaces it
  std::vector<Plot> shown_plots;
  auto drop_old = [&](auto& echoes, double now) {
    echoes.erase(std::remove_if(echoes.begin(), echoes.end(),
                                [&](auto const& echo) { return now - echo.time > detection_lifetime; }),
                 echoes.end());
  };

  // a detection or plot is `range` from the array center, theta from broadside toward the last element
  auto const& first_element = geometry.elements.front();
  auto const& last_element = geometry.elements.back();
  double axis_x = (last_element.x - first_element.x) / geometry.array_length;
  double axis_y = (last_element.y - first_element.y) / geometry.array_length;
  double broadside_x = axis_y;   // the array faces away from its ground plane,
  double broadside_y = -axis_x;  // which is on the left of the first-to-last direction
  double center_x = (first_element.x + last_element.x) / 2.0;
  double center_y = (first_element.y + last_element.y) / 2.0;
  auto to_screen = [&](double out, double along) {
    double pixels_per_meter = static_cast<double>(GetScreenWidth()) / geometry.domain_width;
    return Vector2{
      static_cast<float>((center_x + out * broadside_x + along * axis_x) * pixels_per_meter),
      static_cast<float>((center_y + out * broadside_y + along * axis_y) * pixels_per_meter),
    };
  };
  auto on_screen = [&](auto const& echo) {
    return to_screen(std::cos(echo.theta) * echo.range, std::sin(echo.theta) * echo.range);
  };

  // the latest set of tracks; a set with a new report number replaces it
  std::vector<Track> shown_tracks;
  std::size_t shown_report = 0;

  while (!WindowShouldClose()) {
    // to the sim a dragged target jumps, a frame is only steps_per_frame steps
    double cells_per_pixel = static_cast<double>(grid_width) / GetScreenWidth();
    Vector2 mouse = GetMousePosition();
    auto mouse_x = static_cast<std::ptrdiff_t>(mouse.x * cells_per_pixel);
    auto mouse_y = static_cast<std::ptrdiff_t>(mouse.y * cells_per_pixel);
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
      for (std::size_t i = 0; i < targets.size(); ++i) {
        auto dx = mouse_x - static_cast<std::ptrdiff_t>(targets[i].x);
        auto dy = mouse_y - static_cast<std::ptrdiff_t>(targets[i].y);
        auto radius = static_cast<std::ptrdiff_t>(targets[i].radius);
        if (dx * dx + dy * dy <= radius * radius) dragged = i;
      }
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragged.reset();
    if (dragged) {
      auto& target = targets[*dragged];
      // keep the disk out of the sponge and 1λ in front of the array
      auto radius = static_cast<std::ptrdiff_t>(target.radius);
      auto x = static_cast<std::size_t>(std::clamp(mouse_x,
        static_cast<std::ptrdiff_t>(array_start_x + 20) + radius,
        static_cast<std::ptrdiff_t>(grid_width - border) - radius));
      auto y = static_cast<std::size_t>(std::clamp(mouse_y,
        static_cast<std::ptrdiff_t>(border) + radius,
        static_cast<std::ptrdiff_t>(grid_height - border) - radius));
      if (x != target.x || y != target.y) {
        radar_sim.remove_metal(disk(target.x, target.y, target.radius));
        target.x = x;
        target.y = y;
        // re-add every target, so dragging one across another leaves no hole in it
        for (auto const& other : targets) radar_sim.add_metal(disk(other.x, other.y, other.radius));
        std::copy(features.begin(), features.end(), feature_bytes.begin());
        UpdateTexture(features_tex, feature_bytes.data());
      }
    }

    double start_time = GetTime();
    // requests take effect from the next batch on; the scheduler asks a whole PRI ahead,
    // far more than a batch, so a pulse is never fired after its start
    while (auto request = pulse_requests.pop()) {
      array.fire(*request);
    }
    radar_sim.step(steps_per_frame);
    for (std::size_t s = 0; s < radar_sim.batch_size(); ++s) {
      SampleBatch<n_elements> batch;
      batch.step = radar_sim.batch_start() + s;
      std::copy_n(radar_sim.array_samples(0, s), n_elements, batch.samples.begin());
      // make sure that no batches are dropped and lost
      while(!sample_batches.push(batch)) {};
    }
    double work_time = GetTime() - start_time;
    // a GPU backend hands over its own buffer, a CPU backend's field is uploaded first
    FieldView field = radar_sim.field();
    unsigned ez_buffer = field.gl_buffer;
    if (field.cpu) {
      rlUpdateShaderBuffer(ez_upload_buffer, field.cpu, ez_bytes, 0);
      ez_buffer = ez_upload_buffer;
    }
    BeginDrawing();
    BeginShaderMode(shader);
    SetShaderValueTexture(shader, features_loc, features_tex);  // must be inside BeginShaderMode
    rlBindShaderBuffer(ez_buffer, 0);  // binding 0 in the shader
    DrawTexturePro(quad_tex,
      {0, 0, 1, 1},
      {0, 0, static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight())},
      {0, 0}, 0.0f, WHITE);
    EndShaderMode();

    while (auto detection = detections.pop()) shown_detections.push_back(*detection);
    while (auto plot = plots.pop()) shown_plots.push_back(*plot);
    double now = radar_sim.time();
    drop_old(shown_detections, now);
    drop_old(shown_plots, now);
    for (auto const& detection : shown_detections) {
      auto age = static_cast<float>((now - detection.time) / detection_lifetime);
      DrawCircleV(on_screen(detection), 6.0f, Fade(YELLOW, 1.0f - age));
    }
    for (auto const& plot : shown_plots) {
      Vector2 center = on_screen(plot);
      DrawCircleLines(static_cast<int>(center.x), static_cast<int>(center.y), 14.0f, GREEN);
      DrawText(TextFormat("%zu", plot.detections), static_cast<int>(center.x) + 16, static_cast<int>(center.y) - 8, 16, GREEN);
    }

    while (auto track = tracks.pop()) {
      if (track->report != shown_report) {
        shown_tracks.clear();
        shown_report = track->report;
      }
      shown_tracks.push_back(*track);
    }
    for (auto const& track : shown_tracks) {
      Color color = track.confirmed ? SKYBLUE : GRAY;
      Vector2 center = to_screen(track.out, track.along);
      // the line points to where the track expects the target one scan from now
      Vector2 ahead = to_screen(track.out + track.v_out * detection_lifetime, track.along + track.v_along * detection_lifetime);
      DrawCircleLines(static_cast<int>(center.x), static_cast<int>(center.y), 22.0f, color);
      DrawLineV(center, ahead, color);
      DrawText(TextFormat("#%zu", track.id), static_cast<int>(center.x) - 10, static_cast<int>(center.y) - 42, 16, color);
    }

    DrawText(TextFormat("work %4.2f ms  idle %4.2f ms  fps %d", work_time * 1000.0, (GetFrameTime() - work_time) * 1000.0, GetFPS()), 10, 10, 20, WHITE);
    EndDrawing();
  }
  sample_batches.close();
  worker_thread.join();
  UnloadShader(shader);
  UnloadTexture(features_tex);
  UnloadTexture(quad_tex);
  rlUnloadShaderBuffer(ez_upload_buffer);
  CloseWindow();
  std::printf("closed, samples recorded at dt = %g s\n", radar_sim.dt());
}
