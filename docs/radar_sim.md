# How the radar sim works

An overview of `apps/radar_sim.cpp`: the architecture, the processing chain, and why each part works. The concepts are explained intuitively rather than mathematically.

## 1. The big picture: hardware and computer

The program runs as two halves on two threads, communicating only through queues (the SPSC rings):

- **The sim thread is the physical world plus the antenna hardware.** It steps the electromagnetic field, makes the antenna elements transmit whatever they've been told to, and records what every element receives at every step.
- **The processor thread is the radar's computer.** It decides what to transmit and when, receives the raw samples, finds targets in them, and keeps track of them.

That's how a real radar is split too. The hardware knows nothing about targets; it just transmits on command and digitizes continuously. All the intelligence is in software. The split also keeps the world simulation independent of the processing, and the processing can't "cheat" by looking at the sim's internals. The processor only gets the element positions (`Geometry`) and the samples, as a real radar would.

The queues between them:

| Ring | Direction | Contents |
|---|---|---|
| `pulse_requests` | computer → hardware | "fire this waveform, steered at θ, at time t" |
| `sample_batches` | hardware → computer | every element's reading, for every step |
| `detections`, `plots`, `tracks` | computer → screen | what was found, for drawing |

`main` drives the sim thread: step the grid, push the samples, apply new pulse requests, handle mouse dragging, draw.

## 2. The hardware side

**Field simulation.** `FDTDGrid` steps the electric and magnetic fields cell by cell, which makes waves spread out, reflect off metal and get absorbed by the sponge border. Metal works by forcing the electric field to zero in its cells. Dragging a target used to make the field "explode", because metal only *stopped updating* the field instead of zeroing it. A field frozen at a nonzero value inside the metal acted like a permanent battery that pumped energy into the surroundings every step. `set_pec` now zeroes the field when a cell becomes metal.

**The phased array.** It's a row of 16 elements, each a single cell that both transmits and receives, with a ground plane behind it so it radiates forward.

- **Steering by delays.** Every element sends the same pulse, but slightly delayed relative to its neighbours. If the elements at one end fire a little earlier, the ripples from all of them line up in a slanted wavefront, so the beam leaves at an angle. Changing the delays steers the beam without moving anything. That's the whole trick of a phased array.
- **Pulses are independent.** `fire()` stores each pulse with its own angle and waveform, fixed at the moment of firing. Changing the angle later doesn't affect pulses already in flight, and several pulses can play at once: each element simply outputs the sum. That's how two beams in different directions can be transmitted simultaneously.
- **Receiving.** After every step, the sim writes what each element sees into the array's state. `main` copies that into a `SampleBatch` and pushes it to the processor. The array never knows about the sim or the rings; the sim writes into it, and `main` handles the threading.

## 3. The waveform: why a chirp, and why tapered

A radar wants two conflicting things:

- **Long pulses** carry more energy, so weak echoes are easier to see.
- **Short pulses** let you tell apart two things close together in range.

A **chirp** gets both. The pulse is long, but its frequency sweeps from low to high, so every part of it is distinct from every other part. When processing looks for the exact sweep pattern, the whole long pulse collapses into a sharp, short peak. This is called pulse compression. The wider the sweep (the bandwidth), the sharper the peak. That's why splitting the band in two (section 6) made the range resolution coarser and the peaks weaker.

**The taper.** A pulse that switches on and off abruptly doesn't stay within its band: the hard edges spread energy to nearby frequencies. Ramping the pulse smoothly in and out (the Hann shape, `pulse_taper = 0.5`) keeps its energy in its own band. It costs some energy, but it matters once two pulses share a window (section 6).

## 4. Processing a single pulse

The processor collects the samples from one pulse's fire time until the next fire time. That stretch is the pulse's **window**: one row per step, one column per element. The listen time is chosen so that the farthest echo in the domain gets back before the next pulse. Then:

**a) Receive beamforming (delay and sum).** This is steering in reverse. An echo from direction θ reaches the elements at slightly different times: first the end nearest the target, then the rest. Shifting each element's samples to undo those differences and adding them up makes echoes from θ add up strongly, while echoes from other directions arrive misaligned and partly cancel. The array now "listens" in a direction.

**b) Matched filter.** This slides a copy of the transmitted chirp along the beamformed signal and measures how well they match at every position. Where an echo starts, the match is strong and you get the compressed peak. The comparison is done twice, with the chirp and with a copy shifted by a quarter cycle (I and Q), and the two are combined. A single comparison would rise and fall with the carrier wave; the combination gives a smooth **envelope**, so the peak sits where the echo really is. The reference is rebuilt from each pulse's own waveform, so any waveform a request carries is handled.

**c) Converting time to range.** The time from transmission to the echo, multiplied by the speed of light and halved (out and back), is the distance. Two corrections make it accurate:

- **Measured from the array center.** With steering, the elements fire at different times, so the beam effectively leaves the center a little after the first element fires. That offset is subtracted.
- **Blind range.** While the array is transmitting, and briefly after (until the pulse has traveled across the array to the far elements), the receivers are flooded by the array's own transmission. That early part of the window is skipped. When several pulses share a window, every one of them uses the blind range of the most steered pulse. Leaving out the travel time across the array caused fake detections at about 2 m.

**d) Detection.** Anything in the envelope that rises clearly above the window's average (`detection_factor` times the mean) counts as an echo. Each bump above the threshold becomes one detection, at its highest point.

**e) Angle from the echo.** At first, a detection's angle was just where the beam was pointed, which is accurate only to the beam width (about 8°). Now, for each detection, the processor re-beamforms toward a fan of nearby angles (0.5° apart) and keeps the one where the echo is strongest. A parabola through the best angle and its neighbours refines it further. This measures where the echo actually came from, using the same delay idea as transmit steering in reverse. The transmit beam doesn't bias the result, because it scales every trial angle equally.

## 5. Over several pulses: plots and tracks

**Plots (merging).** The beam is wider than the scan step, so one target is hit by two or three neighbouring pulses, each giving its own detection. At the end of each scan, detections closer together than `merge_distance` are combined into one **plot**, weighted toward the stronger ones. One plot per target per scan.

**Tracks.** A tracker follows targets over time with an alpha-beta filter, in four steps:

1. **Predict:** using the track's last position and velocity, where should the target be now?
2. **Associate:** match each prediction to the nearest plot within the gate (`track_gate`, 1 m), closest pairs first, so no plot feeds two tracks.
3. **Correct:** move the track part of the way toward its plot (alpha). Moving only part of the way smooths out measurement jitter. Adjust the velocity by part of the error (beta): if the target keeps showing up ahead of the prediction, the estimated speed rises.
4. **Bookkeeping:** a plot no track wanted starts a new tentative track. A track becomes confirmed after `confirm_hits` (3) plots, which filters out one-off flukes, and is dropped after more than `max_misses` (3) scans in a row without a plot.

## 6. The scheduler: search and track together

**Search mode** sweeps the beam across the sector, one angle per pulse, and wraps around. A scan is one full sweep.

**Track mode** aims extra pulses at where each confirmed track is predicted to be at fire time. Their echoes correct the track immediately instead of waiting for the end of the scan, so tracks update much more often. A track pulse that misses doesn't count as a miss; only the search scan decides whether a track has disappeared.

**Simultaneous pulses.** Alternating search and track pulses would halve the search speed, so by default (`simultaneous_tracks = true`) each PRI fires both at once in different directions. They share one receive window, and the processor runs the processing chain once for each pulse on the same data.

**Telling them apart.** With two echoes mixed in one window, each pulse's processing must ignore the other's echo. Receive beamforming helps, but not enough, because the track pulse's echo is very strong: it's aimed straight at the target. So the pulses use **different frequency bands**: search 1.2–1.45 GHz, track 1.55–1.8 GHz. The matched filter only responds to its own pulse's chirp pattern, so the other band barely registers. This is where the taper mattered: with abrupt edges, the track pulse spread enough energy into the search band for its strong echo to show up as ghosts in the search pulses. The full Hann taper keeps the bands clean.

## 7. Known weaknesses

- **No noise.** The threshold is relative to the window's average, and in a noise-free sim that average can be almost zero when the beam points at nothing, so tiny leaks can still look significant. Adding receiver noise would give the threshold a real floor and make detection behave like a real radar's.
- **Coarser resolution.** Splitting the band costs range resolution (0.6 m instead of 0.25 m) and some sensitivity, which is why the threshold had to come down to about 8.
- **Dragged targets jump.** A dragged target moves instantly from the sim's point of view, so velocities are huge in sim time and Doppler can't be measured. Tracking from scan to scan works fine.
- **Grating lobes at the top of the band.** At 1.8 GHz, the element spacing (10 cells) is slightly more than half a wavelength. Steered far off broadside, the track band could form a weaker second beam (a grating lobe). It's minor with the current ±35° scan.
