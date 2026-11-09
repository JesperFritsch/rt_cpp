#pragma once
// line plots in the terminal with braille characters, and a fixed area at the top of the
// terminal to redraw them in while other output keeps scrolling below
#include <sys/ioctl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

// printf into a std::string, at most 255 characters; safe to call from any thread
inline std::string format_text(char const* format, ...) __attribute__((format(printf, 1, 2)));
inline std::string format_text(char const* format, ...) {
  char buffer[256];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof buffer, format, args);
  va_end(args);
  return buffer;
}

// a grid of braille characters; each character is 2 dots wide and 4 dots high,
// so a canvas of `cols` x `rows` characters has 2 * cols x 4 * rows dots
class BrailleCanvas {
public:
  BrailleCanvas(std::size_t cols, std::size_t rows) : cols_(cols), rows_(rows), cells_(cols * rows, 0) {}

  std::size_t dot_width() const { return 2 * cols_; }
  std::size_t dot_height() const { return 4 * rows_; }

  // dot (x, y), y = 0 at the top; dots outside the canvas are ignored
  void set(std::size_t x, std::size_t y) {
    if (x >= dot_width() || y >= dot_height()) return;
    // bit of each dot in a braille character, by column and row within it
    static constexpr unsigned char bits[2][4] = {{0x01, 0x02, 0x04, 0x40}, {0x08, 0x10, 0x20, 0x80}};
    cells_[(y / 4) * cols_ + x / 2] |= bits[x % 2][y % 4];
  }

  // line `row` as UTF-8; braille characters are U+2800 plus the dot bits
  std::string row(std::size_t row) const {
    std::string out;
    out.reserve(3 * cols_);
    for (std::size_t c = 0; c < cols_; ++c) {
      unsigned code = 0x2800u + cells_[row * cols_ + c];
      out += static_cast<char>(0xE0u | (code >> 12));
      out += static_cast<char>(0x80u | ((code >> 6) & 0x3Fu));
      out += static_cast<char>(0x80u | (code & 0x3Fu));
    }
    return out;
  }

  std::size_t rows() const { return rows_; }

private:
  std::size_t cols_, rows_;
  std::vector<unsigned char> cells_;
};

// dot row of `value` on a canvas showing y_min at the bottom and y_max at the top
inline std::size_t plot_row(BrailleCanvas const& canvas, float value, float y_min, float y_max) {
  float span = y_max > y_min ? y_max - y_min : 1.0f;
  float from_top = (y_max - value) / span * static_cast<float>(canvas.dot_height() - 1);
  from_top = std::clamp(from_top, 0.0f, static_cast<float>(canvas.dot_height() - 1));
  return static_cast<std::size_t>(from_top + 0.5f);
}

// draws `n` samples across the whole canvas width; each dot column covers a slice of the samples
// and gets a vertical line from their smallest to their largest value, joined to the previous
// column, so a peak shows up however many samples share a column
inline void plot_series(BrailleCanvas& canvas, float const* data, std::size_t n, float y_min, float y_max) {
  if (n == 0) return;
  std::size_t width = canvas.dot_width();
  float previous = data[0];
  for (std::size_t x = 0; x < width; ++x) {
    std::size_t begin = x * n / width;
    std::size_t end = std::max(begin + 1, (x + 1) * n / width);
    float low = previous;
    float high = previous;
    for (std::size_t i = begin; i < end && i < n; ++i) {
      low = std::min(low, data[i]);
      high = std::max(high, data[i]);
    }
    previous = data[std::min(end, n) - 1];
    std::size_t top = plot_row(canvas, high, y_min, y_max);
    std::size_t bottom = plot_row(canvas, low, y_min, y_max);
    for (std::size_t y = top; y <= bottom; ++y) canvas.set(x, y);
  }
}

// a dotted horizontal line at `value`
inline void plot_level(BrailleCanvas& canvas, float value, float y_min, float y_max) {
  std::size_t y = plot_row(canvas, value, y_min, y_max);
  for (std::size_t x = 0; x < canvas.dot_width(); x += 3) canvas.set(x, y);
}

// keeps the top `lines` lines of the terminal for redrawing in place, by limiting scrolling
// to the lines below them; everything else printed scrolls there as usual
// does nothing when stdout is not a terminal or the terminal is too small
class TerminalDashboard {
public:
  explicit TerminalDashboard(std::size_t lines) : lines_(lines) {}

  ~TerminalDashboard() {
    if (rows_ == 0) return;
    // scroll the whole terminal again and leave the cursor below everything
    std::printf("\033[r\033[%zu;1H\n", rows_);
    std::fflush(stdout);
  }

  TerminalDashboard(TerminalDashboard const&) = delete;
  TerminalDashboard& operator=(TerminalDashboard const&) = delete;

  // terminal width in characters, 0 when there is no terminal
  std::size_t width() const {
    winsize size{};
    if (!isatty(STDOUT_FILENO) || ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) != 0) return 0;
    return size.ws_col;
  }

  // replaces the reserved lines with `content`, at most `lines` of them
  void draw(std::vector<std::string> const& content) {
    winsize size{};
    if (!isatty(STDOUT_FILENO) || ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) != 0) return;
    std::size_t rows = size.ws_row;
    if (rows < lines_ + 3) return;  // keep a few lines for the scrolling output
    if (rows != rows_) reserve(rows);

    std::printf("\0337");  // save the cursor, it is somewhere in the scrolling part
    for (std::size_t i = 0; i < lines_; ++i)
      std::printf("\033[%zu;1H\033[2K%s", i + 1, i < content.size() ? content[i].c_str() : "");
    std::printf("\0338");  // back to where the scrolling output continues
    std::fflush(stdout);
  }

private:
  std::size_t lines_;
  std::size_t rows_ = 0;  // terminal height the scroll region was set up for, 0 before the first draw

  void reserve(std::size_t rows) {
    // the first time, push what is on screen up into the scrollback so nothing is drawn over
    if (rows_ == 0)
      for (std::size_t i = 0; i < rows; ++i) std::printf("\n");
    rows_ = rows;
    // scroll only lines_ + 1 .. rows; setting the region moves the cursor, so put it at its bottom
    std::printf("\033[%zu;%zur\033[%zu;1H", lines_ + 1, rows, rows);
  }
};
