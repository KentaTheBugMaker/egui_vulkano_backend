use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chrono::Timelike;
use egui::ClippedMesh;

use epi::App;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::image::ImageUsage;
use vulkano::instance::Instance;
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::GpuFuture;
use vulkano::{swapchain, Version};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow::{Poll, Wait};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use egui_vulkano_backend::{EguiVulkanoBackend, RenderTarget, ScreenDescriptor};
use std::borrow::Borrow;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::device::physical::PhysicalDevice;
use once_cell::sync::{Lazy, OnceCell};

fn main() {
    // The start of this examples is exactly the same as `triangle`. You should read the
    // `triangle` examples if you haven't done so yet.

    let required_extensions = vulkano_win::required_extensions();
    //let instance = Instance::new(None, Version::V1_0, &required_extensions, None).unwrap();
    static instance:OnceCell<Arc<Instance>>=OnceCell::new();
    instance.set(Instance::new(None,Version::V1_0,&required_extensions,None).unwrap()).unwrap();
    let physical = PhysicalDevice::enumerate(instance.get().unwrap()).next().unwrap();

    let event_loop: EventLoop<()> = EventLoop::with_user_event();
    //create surface
    let surface = WindowBuilder::new()
        .with_title("Egui Vulkano Backend sample")
        .build_vk_surface(&event_loop, (instance.get().unwrap().clone()))
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
    let mut egui = EguiVulkanoBackend::new(
        surface.clone(),
        device.clone(),
        queue.clone(),
        swapchain.format(),
    );

    event_loop.run(move |event, _, control_flow| {
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
                let dimensions: [u32; 2] = size.into();
                let (new_swapchain, new_images) =
                    match swapchain.recreate().dimensions(dimensions).build() {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };
                swapchain = new_swapchain;
                egui.resize_frame_buffers(&new_images);
            }
            Event::RedrawEventsCleared => {

                egui.begin_frame(surface.clone());

                let mut quit = false;

                egui::SidePanel::left("my_side_panel").show(egui.ctx(), |ui| {
                    ui.heading("Hello World!");
                    if ui.button("Quit").clicked() {
                        quit = true;
                    }

                    egui::ComboBox::from_label("Version")
                        .width(150.0)
                        .selected_text("foo")
                        .show_ui(ui, |ui| {
                            egui::CollapsingHeader::new("Dev")
                                .default_open(true)
                                .show(ui, |ui| {
                                    ui.label("contents");
                                });
                        });
                });

                let (needs_repaint, shapes) = egui.end_frame(surface.clone());

                *control_flow = if quit {
                    winit::event_loop::ControlFlow::Exit
                } else if needs_repaint {
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

                    let render_target = RenderTarget::FrameBufferIndex(image_num);

                    let mut render_command = AutoCommandBufferBuilder::primary(
                        device.clone(),
                        queue_family,
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();

                    //before egui draw

                    //add egui draw call for command buffer builder
                    egui.paint(render_target, &mut render_command, shapes);

                    //after egui draw

                    egui.painter_mut()
                        .present_to_screen(render_command.build().unwrap(), acquire_future);
                }
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
