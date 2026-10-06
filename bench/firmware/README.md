# Firmware benchmarks

Two firmware images that run the same 1,000 steps of benchmark scenario S2 in `f32`, one with
typed-kalman and one with adskalman. The code and data are identical apart from the filter.
They give the `flash_bytes`, `ram_bytes` and `cycles_per_step` metrics.

## Sizes (no board needed)

```bash
cargo build --release                    # from this directory
../../scripts/firmware_sizes.sh "commit,cpu,os,toolchain,date"   # needs: rustup component add llvm-tools
```

## Cycle counts (needs a board)

The images time the filter loop with the DWT cycle counter and store the cycles per step in the
`CYCLES_PER_STEP` static, then halt at a breakpoint. Tracked in
[#15](https://github.com/joslo2345/typed-kalman/issues/15).

### The default board: NUCLEO-F446RE

`memory.x` is set up for the **ST NUCLEO-F446RE** (STM32F446, Cortex-M4F, 512 KB flash, 128 KB
RAM). Its built-in ST-LINK is the only debugger you need; you also need a USB cable for it
(check the connector on your board revision) and `probe-rs`:

```bash
cargo install probe-rs-tools
cargo build --release
probe-rs run --chip STM32F446RETx target/thumbv7em-none-eabihf/release/fw_typed_kalman
```

Then read `CYCLES_PER_STEP` at the address `nm` gives for it. Printing it over RTT instead is
planned in [#13](https://github.com/joslo2345/typed-kalman/issues/13).

The firmware doesn't configure the clock, so it runs at the 16 MHz reset clock, where flash has
no wait states. Record the clock with the results: at 180 MHz, flash wait states add cycles.

### Other boards

Any board works if it has:

1. **A Cortex-M4F or Cortex-M7** (target `thumbv7em-none-eabihf`). A Cortex-M33 with an FPU
   also works with target `thumbv8m.main-none-eabihf`.
2. **A hardware FPU**, so `f32` runs natively.
3. **A DWT cycle counter.** Cortex-M3, M4, M7 and M33 have one; **M0 and M0+ don't** (for
   example the RP2040 in the original Raspberry Pi Pico).
4. **About 16 KB of flash.** The image is about 14 KB.

| Board | Chip / core | Debugger |
|---|---|---|
| NUCLEO-F446RE | STM32F446, M4F, 180 MHz | built in (the default) |
| NUCLEO-F401RE / F411RE | STM32F401/F411, M4F | built in |
| NUCLEO-G431RB / G474RE | STM32G4, M4F, 170 MHz | built in |
| NUCLEO-L476RG / L432KC | STM32L4, M4F, 80 MHz | built in |
| STM32F407G-DISC1 | STM32F407, M4F, 168 MHz | built in |
| nRF52840 DK | nRF52840, M4F, 64 MHz | built in |
| BBC micro:bit v2 | nRF52833, M4F, 64 MHz | built in |
| FRDM-K64F | Kinetis K64, M4F, 120 MHz | built in |
| NUCLEO-H743ZI2 | STM32H743, M7, 480 MHz | built in; has a double-precision FPU, so `f64` can be measured too |
| NUCLEO-F767ZI | STM32F767, M7, 216 MHz | built in |
| WeAct "Black Pill" F401/F411 | STM32F401/F411, M4F | external probe (ST-LINK or Raspberry Pi Debug Probe) |
| Adafruit Feather/Metro M4 | SAMD51, M4F | external probe |
| Raspberry Pi Pico 2 | RP2350, Cortex-M33 + FPU | external probe; M33 target |

Check the exact part on your board's datasheet: some boards ship in several chip variants. To
switch boards:

- update `memory.x` with the chip's flash and RAM origins and lengths,
- pass its name to `probe-rs` (`probe-rs chip list` shows the names),
- for a Cortex-M33, change the target in `.cargo/config.toml`.

Cycle counts only compare within the same chip. Cortex-M7 parts also have caches, which make
counts vary more between runs, so publish numbers from one popular board, preferably the
NUCLEO-F446RE. On Cortex-M4F boards `f64` is emulated in software, so only `f32` is meaningful
there.
