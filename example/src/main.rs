use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainAcquireFuture, SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::sync::Arc;

use chrono::Timelike;
use egui::FontDefinitions;
use egui_vulkano_backend::ScreenDescriptor;
use egui_winit_platform::{Platform, PlatformDescriptor};
use epi::App;
use std::time::Instant;
use vulkano::image::sys::ImageCreationError::UnsupportedDimensions;

enum EguiUserEvent {
    RequestRedraw,
}

/// This is the repaint signal type that egui needs for requesting a repaint from another thread.
/// It sends the custom RequestRedraw event to the winit event loop.
struct ExampleRepaintSignal(std::sync::Mutex<winit::event_loop::EventLoopProxy<EguiUserEvent>>);

impl epi::RepaintSignal for ExampleRepaintSignal {
    fn request_repaint(&self) {
        self.0
            .lock()
            .unwrap()
            .send_event(EguiUserEvent::RequestRedraw)
            .ok();
    }
}
fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );
    let event_loop = EventLoop::with_user_event();
    let surface = WindowBuilder::new()
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
    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
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
    let repaint_signal = std::sync::Arc::new(ExampleRepaintSignal(std::sync::Mutex::new(
        event_loop.create_proxy(),
    )));

    let mut recreate_swapchain = false;
    //init platform
    let mut platform = Platform::new(PlatformDescriptor {
        physical_width: swapchain.surface().window().inner_size().width,
        physical_height: swapchain.surface().window().inner_size().width,
        scale_factor: swapchain.surface().window().scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    //init egui renderpass
    let mut egui_render_pass = egui_vulkano_backend::EguiVulkanoRenderPass::new(
        device.clone(),
        queue.clone(),
        swapchain.format(),
    );
    let mut demo_app = egui_demo_lib::WrapApp::default();
    let mut previous_frame_time = None;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let start_time = Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            platform.update_time(start_time.elapsed().as_secs_f64());
            if recreate_swapchain {
                let dimensions: [u32; 2] = surface.window().inner_size().into();
                let (new_swapchain, new_images) =
                    match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(ok) => ok,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(err) => panic!("failed to recreate swapchain: {:?}", err),
                    };
                swapchain = new_swapchain;
                images = new_images;
                recreate_swapchain = false;
            }
            // Begin to draw the UI frame.
            let egui_start = Instant::now();
            platform.begin_frame();
            let mut app_output = epi::backend::AppOutput::default();

            let mut frame = epi::backend::FrameBuilder {
                info: epi::IntegrationInfo {
                    web_info: None,
                    cpu_usage: previous_frame_time,
                    seconds_since_midnight: Some(seconds_since_midnight()),
                    native_pixels_per_point: Some(swapchain.surface().window().scale_factor() as _),
                },
                tex_allocator: &mut egui_render_pass,
                output: &mut app_output,
                repaint_signal: repaint_signal.clone(),
            }
            .build();

            // Draw the demo application.
            demo_app.update(&platform.context(), &mut frame);

            // End the UI frame. We could now handle the output and draw the UI with the backend.
            let (_output, paint_commands) = platform.end_frame();
            let paint_jobs = platform.context().tessellate(paint_commands);
            let screen_descriptor = ScreenDescriptor {
                physical_width: swapchain.dimensions()[0],
                physical_height: swapchain.dimensions()[1],
                scale_factor: swapchain.surface().window().scale_factor() as f32,
            };
            //
            let (image_num, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
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
            //prepare
            egui_render_pass.upload_egui_texture(&platform.context().texture());
            egui_render_pass.upload_pending_textures();
            let copy_command=egui_render_pass.update_buffers(&paint_jobs, &screen_descriptor);
            //exec
            let command_buffer = egui_render_pass.execute(
                images[image_num].clone(),
                &paint_jobs,
                &screen_descriptor,
            );

            let frame_time = (Instant::now() - egui_start).as_secs_f64() as f32;
            previous_frame_time = Some(frame_time);

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(),copy_command)
                .unwrap()
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
/// Time of day as seconds since midnight. Used for clock in demo app.
pub fn seconds_since_midnight() -> f64 {
    let time = chrono::Local::now().time();
    time.num_seconds_from_midnight() as f64 + 1e-9 * (time.nanosecond() as f64)
}
