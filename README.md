# egui_vulkano_backend
egui vulkano backend

this crate rendering only.

If you want to use please combine with [egui_winit_platform](https://github.com/hasenbanck/egui_winit_platform).

## known bug 
None
## update 
 * remove vulkano_shader dependency 
 * faster index and vertex buffer allocation
 * skip render glitch mesh (indices or vertices empty)
## Fixed 
 * [can't pass color test ](https://github.com/t18b219k/egui_vulkano_backend/issues/1)
 * change tab in sample at debug build cause crash
## Credit
 * egui_winit_platform developers
 * egui_wgpu_backend developers
  
