#include <rt/gpu_backend.hpp>
#include <rlgl.h>
#include <algorithm>
#include <stdexcept>

// raylib loads OpenGL through glad, which keeps every GL function as a global pointer;
// rlgl wraps glDispatchCompute but not glMemoryBarrier, so use glad's pointer directly
extern "C" {
extern void (*glad_glMemoryBarrier)(unsigned int barriers);
}

namespace {

constexpr unsigned shader_storage_barrier = 0x00002000;  // GL_SHADER_STORAGE_BARRIER_BIT
constexpr unsigned buffer_update_barrier = 0x00000200;   // GL_BUFFER_UPDATE_BARRIER_BIT

constexpr unsigned tile = 16;    // H and E kernels run 16 x 16 cells per workgroup
constexpr unsigned group = 64;   // the 1D kernels run 64 cells per workgroup

// the same updates as CpuBackend, one invocation per cell
// H only reads ez, so hx and hy are updated by the same invocation
constexpr const char* update_h_code = R"(
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer Ez { float ez[]; };
layout(std430, binding = 1) buffer Hx { float hx[]; };
layout(std430, binding = 2) buffer Hy { float hy[]; };
layout(std430, binding = 3) readonly buffer Damp { float damp[]; };
uniform int width;
uniform int height;
uniform float courant;
void main() {
  int x = int(gl_GlobalInvocationID.x);
  int y = int(gl_GlobalInvocationID.y);
  if (x >= width || y >= height) return;  // the last workgroups reach past the grid
  int cell = y * width + x;
  if (y < height - 1) hx[cell] = (hx[cell] - courant * (ez[cell + width] - ez[cell])) * damp[cell];
  if (x < width - 1) hy[cell] = (hy[cell] + courant * (ez[cell + 1] - ez[cell])) * damp[cell];
}
)";

constexpr const char* update_e_code = R"(
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Ez { float ez[]; };
layout(std430, binding = 1) readonly buffer Hx { float hx[]; };
layout(std430, binding = 2) readonly buffer Hy { float hy[]; };
layout(std430, binding = 3) readonly buffer Damp { float damp[]; };
layout(std430, binding = 4) readonly buffer Pec { uint pec[]; };
uniform int width;
uniform int height;
uniform float courant;
void main() {
  int x = int(gl_GlobalInvocationID.x);
  int y = int(gl_GlobalInvocationID.y);
  if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;  // the edge is never updated
  int cell = y * width + x;
  float curl = (hy[cell] - hy[cell - 1]) - (hx[cell] - hx[cell - width]);
  ez[cell] = ez[cell] + (courant * curl) * damp[cell] * float(pec[cell] == 0u);
}
)";

// one invocation per source cell; a cell listed twice would race, so each must appear once
constexpr const char* add_sources_code = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Ez { float ez[]; };
layout(std430, binding = 5) readonly buffer SourceCells { uint source_cells[]; };
layout(std430, binding = 6) readonly buffer SourceValues { float source_values[]; };
uniform int count;  // source cells
uniform int row;    // step within the batch
void main() {
  int i = int(gl_GlobalInvocationID.x);
  if (i >= count) return;
  ez[source_cells[i]] += source_values[row * count + i];
}
)";

constexpr const char* record_probes_code = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Ez { float ez[]; };
layout(std430, binding = 7) readonly buffer ProbeCells { uint probe_cells[]; };
layout(std430, binding = 8) writeonly buffer ProbeOut { float probe_out[]; };
uniform int count;  // probe cells
uniform int row;    // step within the batch
void main() {
  int i = int(gl_GlobalInvocationID.x);
  if (i >= count) return;
  probe_out[row * count + i] = ez[probe_cells[i]];
}
)";

// see CpuBackend::set_pec: ez frozen at a nonzero value in metal would pump energy out forever
constexpr const char* zero_metal_code = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Ez { float ez[]; };
layout(std430, binding = 4) readonly buffer Pec { uint pec[]; };
uniform int cells;
void main() {
  int i = int(gl_GlobalInvocationID.x);
  if (i < cells && pec[i] != 0u) ez[i] = 0.0;
}
)";

unsigned compile_program(char const* code) {
  unsigned shader = rlCompileShader(code, RL_COMPUTE_SHADER);
  if (shader == 0) throw std::runtime_error("GpuBackend: compute shader failed to compile, see the log above");
  unsigned program = rlLoadComputeShaderProgram(shader);
  if (program == 0) throw std::runtime_error("GpuBackend: compute shader failed to link, see the log above");
  return program;
}

void set_int(unsigned program, char const* name, int value) {
  rlEnableShader(program);
  rlSetUniform(rlGetLocationUniform(program, name), &value, RL_SHADER_UNIFORM_INT, 1);
}

void set_float(unsigned program, char const* name, float value) {
  rlEnableShader(program);
  rlSetUniform(rlGetLocationUniform(program, name), &value, RL_SHADER_UNIFORM_FLOAT, 1);
}

unsigned bytes(std::size_t count, std::size_t size) { return static_cast<unsigned>(count * size); }

unsigned groups(std::size_t count, unsigned per_group) {
  return static_cast<unsigned>((count + per_group - 1) / per_group);
}

// a float buffer with room for at least `needed` floats; regrown, and its contents lost, when too small
void ensure_capacity(unsigned& buffer, std::size_t& capacity, std::size_t needed) {
  if (needed <= capacity) return;
  rlUnloadShaderBuffer(buffer);
  capacity = std::max(needed, 2 * capacity);
  buffer = rlLoadShaderBuffer(bytes(capacity, sizeof(float)), nullptr, RL_DYNAMIC_COPY);
}

}  // namespace

GpuBackend::GpuBackend(
  std::size_t width,
  std::size_t height,
  std::size_t border,
  float courant,
  float damp
)
      : width_(width),
        height_(height),
        courant_(courant),
        pec_(width * height, 0) {
  std::size_t cells = width * height;
  // a null data pointer makes rlLoadShaderBuffer clear the buffer to zero
  ez_ = rlLoadShaderBuffer(bytes(cells, sizeof(float)), nullptr, RL_DYNAMIC_COPY);
  hx_ = rlLoadShaderBuffer(bytes(cells, sizeof(float)), nullptr, RL_DYNAMIC_COPY);
  hy_ = rlLoadShaderBuffer(bytes(cells, sizeof(float)), nullptr, RL_DYNAMIC_COPY);
  auto damping = sponge_damping(width, height, border, damp);
  damp_ = rlLoadShaderBuffer(bytes(cells, sizeof(float)), damping.data(), RL_STATIC_DRAW);
  pec_buffer_ = rlLoadShaderBuffer(bytes(cells, sizeof(std::uint32_t)), nullptr, RL_DYNAMIC_DRAW);
  // never empty, so every binding always has a buffer behind it
  source_cells_ = rlLoadShaderBuffer(sizeof(std::uint32_t), nullptr, RL_DYNAMIC_DRAW);
  probe_cells_ = rlLoadShaderBuffer(sizeof(std::uint32_t), nullptr, RL_DYNAMIC_DRAW);
  ensure_capacity(source_values_, source_values_capacity_, 1);
  ensure_capacity(probe_out_, probe_out_capacity_, 1);

  update_h_ = compile_program(update_h_code);
  update_e_ = compile_program(update_e_code);
  add_sources_ = compile_program(add_sources_code);
  record_probes_ = compile_program(record_probes_code);
  zero_metal_ = compile_program(zero_metal_code);

  // uniforms keep their values per program, so the constant ones are set once
  for (unsigned program : {update_h_, update_e_}) {
    set_int(program, "width", static_cast<int>(width_));
    set_int(program, "height", static_cast<int>(height_));
    set_float(program, "courant", courant_);
  }
  set_int(zero_metal_, "cells", static_cast<int>(cells));
  add_sources_row_ = rlGetLocationUniform(add_sources_, "row");
  record_probes_row_ = rlGetLocationUniform(record_probes_, "row");
  rlDisableShader();
}

GpuBackend::~GpuBackend() {
  for (unsigned program : {update_h_, update_e_, add_sources_, record_probes_, zero_metal_})
    rlUnloadShaderProgram(program);
  for (unsigned buffer : {ez_, hx_, hy_, damp_, pec_buffer_, source_cells_, source_values_, probe_cells_, probe_out_})
    rlUnloadShaderBuffer(buffer);
}

void GpuBackend::bind_buffers() const {
  rlBindShaderBuffer(ez_, 0);
  rlBindShaderBuffer(hx_, 1);
  rlBindShaderBuffer(hy_, 2);
  rlBindShaderBuffer(damp_, 3);
  rlBindShaderBuffer(pec_buffer_, 4);
  rlBindShaderBuffer(source_cells_, 5);
  rlBindShaderBuffer(source_values_, 6);
  rlBindShaderBuffer(probe_cells_, 7);
  rlBindShaderBuffer(probe_out_, 8);
}

// replaces `buffer` with one holding the flat index of each cell
unsigned GpuBackend::upload_indices(unsigned buffer, std::vector<Cell> const& cells) {
  std::vector<std::uint32_t> indices;
  indices.reserve(cells.size());
  for (auto const& cell : cells) indices.push_back(static_cast<std::uint32_t>(cell.y * width_ + cell.x));
  if (indices.empty()) indices.push_back(0);  // see the constructor, buffers are never empty
  rlUnloadShaderBuffer(buffer);
  return rlLoadShaderBuffer(bytes(indices.size(), sizeof(std::uint32_t)), indices.data(), RL_DYNAMIC_DRAW);
}

void GpuBackend::set_pec(std::vector<Cell> const& cells, bool metal) {
  for (auto const& cell : cells) pec_[cell.y * width_ + cell.x] = metal ? 1 : 0;
  rlUpdateShaderBuffer(pec_buffer_, pec_.data(), bytes(pec_.size(), sizeof(std::uint32_t)), 0);
  if (!metal) return;
  bind_buffers();
  rlEnableShader(zero_metal_);
  rlComputeShaderDispatch(groups(pec_.size(), group), 1, 1);
  glad_glMemoryBarrier(shader_storage_barrier);
  rlDisableShader();
}

void GpuBackend::set_source_cells(std::vector<Cell> cells) {
  source_count_ = cells.size();
  source_cells_ = upload_indices(source_cells_, cells);
  set_int(add_sources_, "count", static_cast<int>(source_count_));
  rlDisableShader();
}

void GpuBackend::set_probe_cells(std::vector<Cell> cells) {
  probe_count_ = cells.size();
  probe_cells_ = upload_indices(probe_cells_, cells);
  set_int(record_probes_, "count", static_cast<int>(probe_count_));
  rlDisableShader();
}

void GpuBackend::step(std::size_t n, float const* source_values, float* probe_out) {
  if (n == 0) return;
  ensure_capacity(source_values_, source_values_capacity_, n * source_count_);
  ensure_capacity(probe_out_, probe_out_capacity_, n * probe_count_);
  if (source_count_ > 0)
    rlUpdateShaderBuffer(source_values_, source_values, bytes(n * source_count_, sizeof(float)), 0);
  bind_buffers();

  // every dispatch reads what the one before it wrote, so each is followed by a barrier
  for (std::size_t i = 0; i < n; ++i) {
    auto row = static_cast<int>(i);
    if (source_count_ > 0) {
      rlEnableShader(add_sources_);
      rlSetUniform(add_sources_row_, &row, RL_SHADER_UNIFORM_INT, 1);
      rlComputeShaderDispatch(groups(source_count_, group), 1, 1);
      glad_glMemoryBarrier(shader_storage_barrier);
    }
    rlEnableShader(update_h_);
    rlComputeShaderDispatch(groups(width_, tile), groups(height_, tile), 1);
    glad_glMemoryBarrier(shader_storage_barrier);
    rlEnableShader(update_e_);
    rlComputeShaderDispatch(groups(width_, tile), groups(height_, tile), 1);
    glad_glMemoryBarrier(shader_storage_barrier);
    if (probe_count_ > 0) {
      rlEnableShader(record_probes_);
      rlSetUniform(record_probes_row_, &row, RL_SHADER_UNIFORM_INT, 1);
      rlComputeShaderDispatch(groups(probe_count_, group), 1, 1);
      glad_glMemoryBarrier(shader_storage_barrier);
    }
  }
  rlDisableShader();

  // waits for the GPU to finish the batch
  if (probe_count_ > 0) {
    glad_glMemoryBarrier(buffer_update_barrier);
    rlReadShaderBuffer(probe_out_, probe_out, bytes(n * probe_count_, sizeof(float)), 0);
  }
  step_ += n;
}
