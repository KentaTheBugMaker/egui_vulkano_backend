use std::sync::Arc;

use chrono::Timelike;

use vulkano::device::{Device, DeviceExtensions};
use vulkano::image::ImageUsage;
use vulkano::instance::Instance;
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::GpuFuture;
use vulkano::{swapchain, Version};
use vulkano_win::VkSurfaceBuild;

use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use egui_vulkano_backend::EguiVulkanoBackend;

use epi::{App, IntegrationInfo};
use once_cell::sync::OnceCell;
use std::time::Instant;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::device::physical::PhysicalDevice;

enum Event {
    RequestRedraw,
}

struct ExampleRepaintSignal(std::sync::Mutex<winit::event_loop::EventLoopProxy<Event>>);

impl epi::RepaintSignal for ExampleRepaintSignal {
    fn request_repaint(&self) {
        self.0.lock().unwrap().send_event(Event::RequestRedraw).ok();
    }
}

fn main() {
    // The start of this examples is exactly the same as `triangle`. You should read the
    // `triangle` examples if you haven't done so yet.

    let required_extensions = vulkano_win::required_extensions();
    //let instance = Instance::new(None, Version::V1_0, &required_extensions, None).unwrap();
    static INSTANCE: OnceCell<Arc<Instance>> = OnceCell::new();
    INSTANCE
        .set(Instance::new(None, Version::V1_0, &required_extensions, None).unwrap())
        .unwrap();
    let physical = PhysicalDevice::enumerate(INSTANCE.get().unwrap())
        .next()
        .unwrap();

    let event_loop = EventLoop::with_user_event();
    //create surface
    let surface = WindowBuilder::new()
        .with_title("Egui Vulkano Backend sample")
        .build_vk_surface(&event_loop, INSTANCE.get().unwrap().clone())
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
    //create integration
    let mut egui =
        EguiVulkanoBackend::new(surface.clone(), device.clone(), queue, swapchain.format());
    egui.create_frame_buffers(&images);
    //restricted  eframe setup
    let mut app_output = epi::backend::AppOutput::default();

    let repaint_signal = std::sync::Arc::new(ExampleRepaintSignal(std::sync::Mutex::new(
        event_loop.create_proxy(),
    )));

    let mut previous_frame_time = None;
    let mut demo_app = egui_demo_lib::WrapApp::default();
    event_loop.run(move |event, _, control_flow| {
        let mut redraw = || {
            egui.begin_frame();
            let ctx = egui.ctx().clone();
            let mut frame = epi::backend::FrameBuilder {
                info: IntegrationInfo {
                    web_info: None,
                    prefer_dark_mode: None,
                    cpu_usage: previous_frame_time,
                    seconds_since_midnight: Some(seconds_since_midnight()),
                    native_pixels_per_point: Some(egui.pixels_per_point()),
                },
                tex_allocator: &mut egui,
                output: &mut app_output,
                repaint_signal: repaint_signal.clone(),
            }
            .build();

            let egui_start = Instant::now();

            demo_app.update(&ctx, &mut frame);

            let (needs_repaint, shapes) = egui.end_frame();
            let frame_time = (Instant::now() - egui_start).as_secs_f64() as f32;
            previous_frame_time = Some(frame_time);
            *control_flow = if needs_repaint {
                surface.window().request_redraw();
                winit::event_loop::ControlFlow::Poll
            } else {
                winit::event_loop::ControlFlow::Wait
            };

            {
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
                let mut render_command = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue_family,
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                //add egui draw call for command buffer builder
                egui.paint(image_num, &mut render_command, shapes);
                egui.painter_mut()
                    .present_to_screen(render_command.build().unwrap(), acquire_future);
            }
        };
        match event {
            // Platform-dependent event handlers to workaround a winit bug
            // See: https://github.com/rust-windowing/winit/issues/987
            // See: https://github.com/rust-windowing/winit/issues/1619
            winit::event::Event::RedrawEventsCleared if cfg!(windows) => redraw(),
            winit::event::Event::RedrawRequested(_) if !cfg!(windows) => redraw(),

            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::Resized(size),
                ..
            } => {
                let dimensions: [u32; 2] = size.into();
                let (new_swapchain, new_images) =
                    match swapchain.recreate().dimensions(dimensions).build() {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };
                swapchain = new_swapchain;
                egui.create_frame_buffers(&new_images);
            }
            winit::event::Event::WindowEvent { event, .. } => {
                if egui.is_quit_event(&event) {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }

                egui.on_event(&event);

                surface.window().request_redraw(); // TODO: ask egui if the events warrants a repaint instead
            }

            _ => (),
        }
    });
}

/// Time of day as seconds since midnight. Used for clock in demo app.
pub fn seconds_since_midnight() -> f64 {
    let time = chrono::Local::now().time();
    time.num_seconds_from_midnight() as f64 + 1e-9 * (time.nanosecond() as f64)
}
