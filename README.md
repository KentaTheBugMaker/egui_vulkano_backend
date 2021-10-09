# egui_vulkano_backend

[![Latest version](https://img.shields.io/crates/v/egui_vulkano_backend.svg)](https://crates.io/crates/egui_vulkano_backend)
[![Documentation](https://docs.rs/egui_vulkano_backend/badge.svg)](https://docs.rs/egui_vulkano_backend)
[![Build Status](https://github.com/t18b219k/egui_vulkano_backend/workflows/CI/badge.svg)](https://github.com/t18b219k/egui_vulkano_backend/actions?workflow=CI)
[![dependency status](https://deps.rs/repo/github/t18b219k/egui_vulkano_backend/status.svg)](https://deps.rs/repo/github/t18b219k/egui_vulkano_backend)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

Backend code to run [egui](https://crates.io/crates/egui) using [vulkano](https://crates.io/crates/vulkano).


## sample

[bit complex example](https://github.com/t18b219k/egui_vulkano_backend/tree/master/examples/off_screen/main.rs)

```shell
cargo run --example off_screen
```

We have created [a simple example](https://github.com/t18b219k/egui_vulkano_backend/tree/master/examples/demo.rs)
project to show you, how to use this crate.

```shell
cargo run --example demo
```
## Known Issue

## Not Released
* target egui future release + vulkano 0.26
* demo changed
* use official integration `egui_for_winit` like `egui_glium` does
## Update v0.14.0 ([Affected version](#stack_overflow_on_debug_build))
* target egui 0.14. + vulkano 0.25
* remove wait_image_upload
* demo change

## Update v0.6.0
* parallel buffer upload
* remove iter_vec dependency
* proper clipping
## Update v0.5.0

* target egui 0.12.0 + vulkano 0.24.0
* no change in this crate API, but you need to work for vulkano change

## Update v0.4.1(0f00641)

* removed egui runner

## Update v0.4.0

* target egui 0.12.0 + vulkano 0.23
* api breaking change

## Update v0.3.0

* add egui runner
* api breaking change see [port guide](port_guide_v030.md)
* add new function that support recreating and initializing render area
* target vulkano 0.22.0

## Update v0.2.2

* remove bytemuck dependency

## Update v0.2.1

* remove bug screenshot

## Update v0.2.0

* rename api
    * upload_egui_texture -> request_upload_egui_texture
    * upload_pending_textures -> wait_texture_upload
* nonblocking image upload
* remove temporary index and vertex alloc
* remove uniform buffer

## Update v0.1.0

* reduce uniform buffer and descriptor set allocation (v0.1.0)
* reduce index and vertex buffer allocation (v0.1.0)

## Update

* remove vulkano_shader dependency extremely faster build time
* faster index and vertex buffer allocation
* skip render glitch mesh (index or vertices empty)

## Fixed

* doesn't pass color test
* change tab in sample at debug build cause crash.
* in egui_demo color test scrollbar glitch 
* [stack over flow on debug build](#stack_overflow_on_debug_build)
## Version list

|egui_vulkano_backend|egui |vulkano |vulkano-shader(dependency) |vulkano-win(if use runner)|
|-----|------|------|------|---|
|0.0.1|0.10.0|0.20.0|0.20.0|not support|
|0.0.2|0.10.0|0.20.0|0.20.0|not support|
|0.0.3|0.10.0|0.21.0|0.21.0|not support|
|0.0.4|0.10.0|0.21.0|none|not support|
|0.0.5|0.10.0|0.21.0|none|not support|
|0.1.0|0.10.0|0.21.0|none|not support|
|0.2.0|0.10.0|0.21.0|none|not support|
|0.2.1|0.10.0|0.21.0|none|not support|
|0.2.2|0.10.0|0.21.0|none|not support|
|0.3.0|0.10.0|0.22.0|none|0.22.0|
|0.4.0|0.12.0|0.23.0|none|0.23.0|
|0.4.1|0.12.0|0.23.0|none|removed|
|0.5.0|0.12.0|0.24.0|none|removed|
|0.5.0|0.13|0.24|none|removed|
|0.6.0|0.13|0.24|none|removed|
|0.14.0|0.14.0|0.25.0|none|removed|
## License

egui_vulkano_backend is distributed under the terms of both the MIT license, and the Apache License (Version 2.0).
See [LICENSE-APACHE](https://github.com/t18b219k/egui_vulkano_backend/blob/master/LICENSE-APACHE)
, [LICENSE-MIT](https://github.com/t18b219k/egui_vulkano_backend/blob/master/LICENSE-MIT).

## Thanks

* egui_wgpu_backend developers
* egui_winit_platform developers
* bug reporter
## stack_overflow_on_debug_build

stack overflow on debug build from 0.14.0 (bisected) on Windows
  * I tested official vulkano-shaders shader! macro produce same result.
  * if you use release build not affected
  * vulkano 0.25 related problem maybe they can't read properly shader that have push constants. 