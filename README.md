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
 * reduce crate size(v0.2.3)
 * remove vulkano_shader dependency extremely faster build time 
 * faster index and vertex buffer allocation
 * skip render glitch mesh (index or vertices empty)
 * rename api(v0.2.0)
   * upload_egui_texture -> request_upload_egui_texture
   * upload_pending_textures -> wait_texture_upload 
 * nonblocking image upload(v0.2.0) 
 * remove temporary index and vertex alloc(v0.2.0)  
 * skip render 0 size mesh(v0.2.0) 
 * reduce uniform buffer and descriptor set allocation (v0.1.0)
 * reduce  index and vertex buffer allocation (v0.1.0)
## Fixed
 * doesn't pass color test
 * change tab in sample at debug build cause crash.
## Example
We have created [a simple example](https://github.com/t18b219k/egui_vulkano_backend/tree/master/examples/demo.rs) project to show you, how to use this crate.
```shell
cargo run --example demo
```
## License
egui_vulkano_backend is distributed under the terms of both the MIT license, and the Apache License (Version 2.0).

See [LICENSE-APACHE](https://github.com/t18b219k/egui_vulkano_backend/blob/master/LICENSE-APACHE), [LICENSE-MIT](https://github.com/t18b219k/egui_vulkano_backend/blob/master/LICENSE-MIT).
## Credits
 * egui_wgpu_backend developers
 * egui_winit_platform developers
