# egui_vulkano_backend

[![Latest version](https://img.shields.io/crates/v/egui_vulkano_backend.svg)](https://crates.io/crates/egui_vulkano_backend)
[![Documentation](https://docs.rs/egui_vulkano_backend/badge.svg)](https://docs.rs/egui_vulkano_backend)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

Backend code to run egui using vulkano.
## known bug
 * When change tab in sample at debug build cause crash
caused from vulkano debug assert so please build in release. 
## Fixed
 * [can't pass color test ](https://github.com/t18b219k/egui_vulkano_backend/issues/1)
 * [![bug screen shot]](https://github.com/t18b219k/egui_vulkano_backend/blob/master/Screenshot%20from%202021-03-09%2023-48-42.png)
## Example
We have created [a simple example](https://github.com/t18b219k/egui_vulkano_backend/tree/master/example) project to show you, how to use this crate.

## License
egui_vulkano_backend is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT).
## Credits
    * egui_wgpu_backend developpers
    * egui_wgpu_platform developpers
