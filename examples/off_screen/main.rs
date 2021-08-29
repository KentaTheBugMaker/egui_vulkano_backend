use std::borrow::Borrow;
use std::time::Instant;

use egui::FontDefinitions;
use egui_winit_platform::{Platform, PlatformDescriptor};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::image::ImageUsage;
use vulkano::instance::Instance;
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::GpuFuture;
use vulkano::{swapchain, Version};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use egui_vulkano_backend::{RenderTarget, ScreenDescriptor};

use crate::renderer::TeapotRenderer;
use vulkano::device::physical::PhysicalDevice;

mod model;
mod renderer;

fn main() {
    // The start of this examples is exactly the same as `triangle`. You should read the
    // `triangle` examples if you haven't done so yet.

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(
        None,
        Version {
            major: 1,
            minor: 2,
            patch: 0,
        },
        &required_extensions,
        None,
    )
    .unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.properties().device_name,
        physical.properties().device_type
    );

    let event_loop: EventLoop<()> = EventLoop::with_user_event();
    let surface = WindowBuilder::new()
        .with_title("Egui Vulkano Backend sample")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        assert!(caps.supported_formats.contains(&(
            vulkano::format::Format::R8G8B8A8Srgb,
            vulkano::swapchain::ColorSpace::SrgbNonLinear
        )));
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::start(device.clone(), surface.clone())
            .format(vulkano::format::Format::R8G8B8A8Srgb)
            .dimensions(dimensions)
            .composite_alpha(alpha)
            .num_images(caps.min_image_count)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .clipped(true)
            .color_space(vulkano::swapchain::ColorSpace::SrgbNonLinear)
            .build()
            .unwrap()
    };

    //create renderer
    let mut egui_render_pass = egui_vulkano_backend::EguiVulkanoRenderPass::new(
        device.clone(),
        queue.clone(),
        swapchain.format(),
    );
    egui_render_pass.create_frame_buffers(images.as_slice());
    //init egui
    // create relation between TextureID and render target
    let (texture_id, mut render_target) = egui_render_pass
        .init_vulkano_image_with_dimensions([1280, 720])
        .unwrap();
    let size = surface.window().inner_size();
    // We use the egui_winit_platform crate as the platform.
    let mut platform = Platform::new(PlatformDescriptor {
        physical_width: size.width,
        physical_height: size.height,
        scale_factor: surface.window().scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });
    let start_time = Instant::now();
    //create renderer
    let mut teapot_renderer = TeapotRenderer::new(device.clone(), queue);
    //set render target
    teapot_renderer.set_render_target(render_target.clone());
    let mut screen_descriptor = ScreenDescriptor {
        physical_width: size.width,
        physical_height: size.height,
        scale_factor: surface.window().scale_factor() as f32,
    };
    let mut rotate = 0.0;
    let mut height_percent = 0.7;
    let mut image_size = [size.width as f32, size.height as f32 * height_percent];
    event_loop.run(move |event, _, control_flow| {
        // Pass the winit events to the platform integration.
        platform.handle_event(&event);
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                match swapchain.recreate().dimensions(size.into()).build() {
                    Ok(r) => {
                        swapchain = r.0;
                        egui_render_pass.create_frame_buffers(&r.1);
                    }
                    Err(SwapchainCreationError::UnsupportedDimensions) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                }
                //resize render_target
                render_target = egui_render_pass
                    .recreate_vulkano_texture_with_dimensions(
                        texture_id,
                        [size.width, (size.height as f32 * height_percent) as u32],
                    )
                    .unwrap();
                image_size = [size.width as f32, size.height as f32 * height_percent];
                //reset framebuffer
                teapot_renderer.set_render_target(render_target.clone());
                //set screen descriptor
                screen_descriptor.physical_height = size.height;
                screen_descriptor.physical_width = size.width;
            }
            Event::RedrawEventsCleared => {
                platform.update_time(start_time.elapsed().as_secs_f64());

                // Begin to draw the UI frame.

                platform.begin_frame();

                egui::CentralPanel::default().show(platform.context().borrow(), |ui| {
                    ui.vertical(|ui| {
                        ui.image(texture_id, image_size);
                        ui.horizontal(|ui| {
                            //add Model view adjuster
                            if ui
                                .add(egui::widgets::Slider::new(&mut height_percent, 0.1..=0.9))
                                .changed()
                            {
                                let size = surface.window().inner_size();

                                render_target = egui_render_pass
                                    .recreate_vulkano_texture_with_dimensions(
                                        texture_id,
                                        [size.width, (size.height as f32 * height_percent) as u32],
                                    )
                                    .unwrap();
                                teapot_renderer.set_render_target(render_target.clone());
                                image_size =
                                    [size.width as f32, size.height as f32 * height_percent];
                            }
                            //add rotation control
                            if ui
                                .add(egui::Slider::new(
                                    &mut rotate,
                                    -std::f32::consts::PI..=std::f32::consts::PI,
                                ))
                                .changed()
                            {
                                teapot_renderer.set_rotate(rotate)
                            }
                        });
                    })
                });
                // End the UI frame. We could now handle the output and draw the UI with the backend.
                let (_output, paint_commands) = platform.end_frame(None);
                let paint_jobs = platform.context().tessellate(paint_commands);

                let mut previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    return;
                }
                teapot_renderer.draw();
                egui_render_pass.request_upload_egui_texture(&platform.context().texture());

                let render_target = RenderTarget::FrameBufferIndex(image_num);
                let render_command = egui_render_pass.create_command_buffer(
                    render_target,
                    paint_jobs,
                    &screen_descriptor,
                );

                egui_render_pass.present_to_screen(render_command, acquire_future);
                *control_flow = ControlFlow::Poll
            }
            _ => (),
        }
    });
}
