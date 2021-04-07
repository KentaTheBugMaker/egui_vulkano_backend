//! io/render egui integration  for vulkano
//!
use crate::{EguiVulkanoRenderPass, RenderTarget, ScreenDescriptor};
use chrono::Timelike;
use egui_winit_platform::{Platform, PlatformDescriptor};
use epi::backend::AppOutput;
use epi::IntegrationInfo;

use std::time::Instant;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::image::ImageUsage;
use vulkano::instance::PhysicalDevice;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano_win::VkSurfaceBuild;
use winit::event::WindowEvent;
use winit::event_loop::ControlFlow;
use winit::window::WindowBuilder;

struct RequestRepaintEvent;
struct VulkanoRepaintSignal(
    std::sync::Mutex<winit::event_loop::EventLoopProxy<RequestRepaintEvent>>,
);
impl epi::RepaintSignal for VulkanoRepaintSignal {
    fn request_repaint(&self) {
        self.0.lock().unwrap().send_event(RequestRepaintEvent).ok();
    }
}
pub fn run(mut app: Box<dyn epi::App>) {
    let event_loop = winit::event_loop::EventLoop::with_user_event();
    let required_extensions = vulkano_win::required_extensions();
    let instance = vulkano::instance::Instance::new(
        Some(&vulkano::app_info_from_cargo_toml!()),
        &required_extensions,
        None,
    )
    .expect("Failed to get instance ");
    let physical_device = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("Failed to get physical device");
    let surface = WindowBuilder::new()
        .with_title(app.name())
        .build_vk_surface(&event_loop, instance.clone())
        .expect("Failed to create surface");
    let queue_family = physical_device
        .queue_families()
        .find(|q| q.supports_graphics() && surface.is_supported(*q).unwrap_or(false))
        .unwrap();
    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical_device,
        physical_device.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        //     let alpha = CompositeAlpha::PreMultiplied;
        assert!(&caps
            .supported_formats
            .contains(&(Format::B8G8R8A8Srgb, ColorSpace::SrgbNonLinear)));
        let format = Format::B8G8R8A8Srgb;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .unwrap()
    };
    let repaint_signal = std::sync::Arc::new(VulkanoRepaintSignal(std::sync::Mutex::new(
        event_loop.create_proxy(),
    )));
    let mut previous_frame_time = None;
    let mut recreate_swapchain = false;
    let mut egui_render_pass = EguiVulkanoRenderPass::new(device, queue, swapchain.format());
    egui_render_pass.create_frame_buffers(&images);
    let start_time = Instant::now();
    #[cfg(feature = "runner_http")]
    let http = std::sync::Arc::new(epi_http::EpiHttp {});

    let mut platform = Platform::new(PlatformDescriptor {
        physical_width: surface.window().inner_size().width,
        physical_height: surface.window().inner_size().height,
        scale_factor: surface.window().scale_factor(),
        font_definitions: Default::default(),
        style: Default::default(),
    });
    event_loop.run(move |event, _, control_flow| {
        platform.handle_event(&event);
        let mut redraw = || {
            platform.update_time(start_time.elapsed().as_secs_f64());

            let (image_num, suboptimal, acquire_future) =
                match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };
            if suboptimal {
                recreate_swapchain = true;
            }
            let screen_descriptor = ScreenDescriptor {
                physical_width: surface.window().inner_size().width,
                physical_height: surface.window().inner_size().height,
                scale_factor: surface.window().scale_factor() as f32,
            };
            let pixel_per_point = surface.window().scale_factor();
            // egui start
            let frame_start = Instant::now();
            platform.begin_frame();
            let mut app_output = epi::backend::AppOutput::default();
            let mut frame = epi::backend::FrameBuilder {
                info: IntegrationInfo {
                    web_info: None,
                    cpu_usage: previous_frame_time,
                    seconds_since_midnight: Some(seconds_since_midnight()),
                    native_pixels_per_point: Some(pixel_per_point as _),
                },
                tex_allocator: &mut egui_render_pass,
                #[cfg(feature = "runner_http")]
                http: http.clone(),
                output: &mut app_output,
                repaint_signal: repaint_signal.clone(),
            }
            .build();
            app.update(&platform.context(), &mut frame);
            let (egui_output, shapes) = platform.end_frame();
            let clipped_meshes = platform.context().tessellate(shapes);
            let frame_time = (Instant::now() - frame_start).as_secs_f32();
            previous_frame_time = Some(frame_time);
            // egui end
            egui_render_pass.request_upload_egui_texture(&platform.context().texture());
            let render_target = RenderTarget::FrameBufferIndex(image_num);
            let render_command = egui_render_pass.create_command_buffer(
                render_target,
                &clipped_meshes,
                &screen_descriptor,
            );
            egui_render_pass.present_to_screen(render_command, acquire_future);
            {
                let AppOutput { quit, window_size } = app_output;
                if let Some(window_size) = window_size {
                    surface.window().set_inner_size(
                        winit::dpi::PhysicalSize {
                            width: (platform.context().pixels_per_point() * window_size.x).round(),
                            height: (platform.context().pixels_per_point() * window_size.y).round(),
                        }
                        .to_logical::<f32>(surface.window().scale_factor()),
                    );
                }
                *control_flow = if quit {
                    winit::event_loop::ControlFlow::Exit
                } else if egui_output.needs_repaint {
                    surface.window().request_redraw();
                    winit::event_loop::ControlFlow::Poll
                } else {
                    winit::event_loop::ControlFlow::Wait
                }
            }
        };
        match event {
            winit::event::Event::RedrawEventsCleared if cfg!(windows) => redraw(),
            winit::event::Event::RedrawRequested(_) if !cfg!(windows) => redraw(),
            winit::event::Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                let dimensions: [u32; 2] = surface.window().inner_size().into();
                let (new_swapchain, new_images) =
                    match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };
                swapchain = new_swapchain;
                egui_render_pass.create_frame_buffers(&new_images);
            }
            winit::event::Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            winit::event::Event::MainEventsCleared
            | winit::event::Event::UserEvent(RequestRepaintEvent) => {
                surface.window().request_redraw()
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
