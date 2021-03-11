# egui_vulkano_backend

[![Latest version](https://img.shields.io/crates/v/egui_vulkano_backend.svg)](https://crates.io/crates/egui_vulkano_backend)
[![Documentation](https://docs.rs/egui_vulkano_backend/badge.svg)](https://docs.rs/egui_vulkano_backend)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

Backend code to run [egui](https://crates.io/crates/egui) using [vulkano](https://crates.io/crates/vulkano).

this crate rendering only you need I/O egui integration e.g. [egui_winit_platform](https://crates.io/crates/egui_winit_platform) 
## known bug 
None
## Update 
 * remove vulkano_shader dependency extremely faster build time 
 * faster index and vertex buffer allocation
 * Skip render glitch mesh (index or vertices empty)
## Fixed
 * [can't pass color test ](https://github.com/t18b219k/egui_vulkano_backend/issues/1)
 * When change tab in sample at debug build cause crash
## Example
We have created [a simple example](https://github.com/t18b219k/egui_vulkano_backend/tree/master/example) project to show you, how to use this crate.

## License
egui_vulkano_backend is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT).
## Credits
    * egui_wgpu_backend developpers
    * egui_wgpu_platform developpers
