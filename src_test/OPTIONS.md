# TOML ini file options

All executables take a single argument: the case name. Settings are read from `case_name.ini`, input from `case_name_input.nc`, output written to `case_name_output.nc`.

## Executables

| Short name | Executable                | Source file             |
|------------|---------------------------|-------------------------|
| **cpp**    | `test_rte_rrtmgp`         | `test_rte_rrtmgp.cpp`   |
| **gpu**    | `test_rte_rrtmgp_gpu`     | `test_rte_rrtmgp.cu`    |
| **bw**     | `test_rte_rrtmgp_bw_gpu`  | `test_rte_rrtmgp_bw.cu` |
| **rt**     | `test_rte_rrtmgp_rt_gpu`  | `test_rte_rrtmgp_rt.cu` |
| **lite**   | `test_rt_lite`            | `test_rt_lite.cu`        |

## `[switches]` (bool)

| Key                    | Default | cpp | gpu | bw  | rt  | lite |
|------------------------|---------|-----|-----|-----|-----|------|
| `shortwave`            | `true`  | x   | x   | x   | x   |      |
| `longwave`             | `true`  | x   | x   |     | x   |      |
| `longwave`             | `false` |     |     | x   |     |      |
| `fluxes`               | `true`  | x   | x   | x   | x   |      |
| `cloud-optics`         | `false` | x   | x   | x   | x   |      |
| `aerosol-optics`       | `false` | x   | x   | x   | x   |      |
| `output-optical`       | `false` | x   | x   | x   |     |      |
| `output-bnd-fluxes`    | `false` | x   | x   | x   |     |      |
| `delta-cloud`          | `true`  | x   | x   |     |     |      |
| `delta-cloud`          | `false` |     |     | x   | x   |      |
| `delta-aerosol`        | `false` | x   | x   | x   | x   |      |
| `timings`              | `false` |     | x   |     |     |      |
| `raytracing`           | `true`  |     |     | x   |     | x    |
| `bw-raytracing`        | `true`  |     |     |     |     | x    |
| `two-stream`           | `true`  |     |     |     |     | x    |
| `liq-cloud-optics`     | `false` |     |     | x   | x   |      |
| `ice-cloud-optics`     | `false` |     |     | x   | x   |      |
| `cloud-mie`            | `false` |     |     | x   | x   | x    |
| `lu-albedo`            | `false` |     |     | x   |     |      |
| `image`                | `true`  |     |     | x   |     |      |
| `broadband`            | `false` |     |     | x   |     |      |
| `profiling`            | `false` |     |     | x   | x   | x    |
| `cloud-cam`            | `false` |     |     | x   |     |      |
| `sw-two-stream`        | `false` |     |     |     | x   |      |
| `sw-raytracing`        | `true`  |     |     |     | x   |      |
| `lw-raytracing`        | `true`  |     |     |     | x   |      |
| `independent-column`   | `false` |     |     |     | x   | x    |
| `lw-scattering`        | `false` |     |     |     | x   |      |
| `single-gpt`           | `false` |     |     |     | x   |      |
| `min-mfp-grid-ratio`   | `true`  |     |     |     | x   |      |
| `tica`                 | `false` |     |     |     | x   |      |

## `[ints]`

| Key                  | Type  | Default | Executables |
|----------------------|-------|---------|-------------|
| `photons-per-pixel`  | `int` | `1`     | bw          |
| `sw-raytracing`      | `Int` | `256`   | rt          |
| `lw-raytracing`      | `Int` | `22`    | rt          |
| `single-gpt`         | `int` | `1`     | rt          |
| `raytracing`         | `Int` | `32`    | lite        |
| `bw-raytracing`      | `Int` | `32`    | lite        |

## `[floats]`

| Key                  | Type    | Default | Executables |
|----------------------|---------|---------|-------------|
| `min-mfp-grid-ratio` | `Float` | `1.0`   | rt          |

## Example ini file

```toml
[switches]
shortwave = true
longwave = true
cloud-optics = true
delta-cloud = true

[ints]
sw-raytracing = 128

[floats]
min-mfp-grid-ratio = 0.5
```
