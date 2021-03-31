//! [egui](https://docs.rs/egui) rendering integration for vulkano
//!
//! This crate have some features
//! * background user texture upload
//! * map vulkano texture to user texture
//! * [easily extend to multi thread rendering](https://github.com/t18b219k/egui_vulkano_backend/blob/master/examples/multi_thread_rendering.rs)
//! * faster than official backend
//! * lower cpu usage
//! * lower memory usage
mod render_pass;
mod shader;

extern crate vulkano;

use crate::render_pass::EguiRenderPassDesc;
use epi::egui;
use epi::egui::{ClippedMesh, Color32, Texture, TextureId};

use std::sync::{Arc, Mutex};
use vulkano::buffer::{BufferSlice, BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{
    AutoCommandBuffer, CommandBufferExecFuture, DynamicState, SubpassContents,
};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::{DescriptorSet, PipelineLayoutAbstract};
use vulkano::device::{Device, Queue};
use vulkano::format::Format::R8G8B8A8Srgb;
use vulkano::framebuffer::{FramebufferAbstract, RenderPass};
use vulkano::image::{Dimensions, ImageViewAccess, ImmutableImage, MipmapsCount, SwapchainImage};

use crate::shader::create_pipeline;
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::framebuffer::Framebuffer;

use iter_vec::ExactSizedIterVec;
use std::sync::mpsc::{Receiver, Sender};

use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::{Scissor, Viewport};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainAcquireFuture};
use vulkano::sync::{GpuFuture, NowFuture};

#[derive(Default, Debug, Copy, Clone)]
struct EguiVulkanoVertex {
    a_pos: [f32; 2],
    a_tex_coord: [f32; 2],
    a_color: u32,
}

struct IsEguiTextureMarker(u32);
const IS_EGUI: IsEguiTextureMarker = IsEguiTextureMarker(1);
const IS_USER: IsEguiTextureMarker = IsEguiTextureMarker(0);
#[repr(C)]
struct PushConstants {
    u_screen_size: [f32; 2],
    marker: IsEguiTextureMarker,
}
vulkano::impl_vertex!(EguiVulkanoVertex, a_pos, a_tex_coord, a_color);
/// same as [egui_wgpu_backend::ScreenDescriptor](https://docs.rs/egui_wgpu_backend/0.5.0/egui_wgpu_backend/struct.ScreenDescriptor.html)
pub struct ScreenDescriptor {
    /// Width of the window in physical pixel.
    pub physical_width: u32,
    /// Height of the window in physical pixel.
    pub physical_height: u32,
    /// HiDPI scale factor.
    pub scale_factor: f32,
}
impl ScreenDescriptor {
    fn logical_size(&self) -> (u32, u32) {
        let logical_width = self.physical_width as f32 / self.scale_factor;
        let logical_height = self.physical_height as f32 / self.scale_factor;
        (logical_width as u32, logical_height as u32)
    }
}
type Pipeline = GraphicsPipeline<
    SingleBufferDefinition<EguiVulkanoVertex>,
    Box<dyn PipelineLayoutAbstract + Send + Sync>,
    Arc<RenderPass<EguiRenderPassDesc>>,
>;
/// egui rendering command builder
pub struct EguiVulkanoRenderPass {
    pipeline: Arc<Pipeline>,
    egui_texture_descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
    egui_texture_version: Option<u64>,
    user_textures: Vec<Option<UserTexture>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    frame_buffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    vertex_buffer_pool: CpuBufferPool<EguiVulkanoVertex>,
    index_buffer_pool: CpuBufferPool<u32>,
    image_staging_buffer: CpuBufferPool<u8>,
    descriptor_set_0: Arc<dyn DescriptorSet + Send + Sync>,
    dynamic: DynamicState,
    image_transfer_request_list: Vec<TextureId>,
    request_sender: Sender<Work>,
    done_notifier: Receiver<TextureId>,
}
type Work = (
    TextureId,
    CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
);
fn spawn_image_upload_thread() -> (Sender<Work>, Receiver<TextureId>) {
    let (tx, rx): (Sender<Work>, Receiver<Work>) = std::sync::mpsc::channel();
    let (notifier, getter) = std::sync::mpsc::channel();
    let rx = Mutex::new(rx);
    std::thread::spawn(move || {
        for request in rx.lock().unwrap().iter() {
            request.1.flush().unwrap();
            notifier.send(request.0).unwrap();
        }
    });
    (tx, getter)
}
impl EguiVulkanoRenderPass {
    ///create command builder
    ///
    /// if render target format incompatible with SwapChain format may cause color glitch
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_target_format: vulkano::format::Format,
    ) -> Self {
        let pipeline = create_pipeline(device.clone(), render_target_format);
        let sampler = Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Linear,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        .unwrap();

        let vertex_buffer_pool = CpuBufferPool::vertex_buffer(device.clone());
        let index_buffer_pool = CpuBufferPool::new(device.clone(), BufferUsage::index_buffer());
        let image_staging_buffer =
            CpuBufferPool::new(device.clone(), BufferUsage::transfer_source());
        let (request_sender, done_notifier) = spawn_image_upload_thread();
        let descriptor_set_0 = Arc::new(
            PersistentDescriptorSet::start(
                pipeline.layout().descriptor_set_layout(0).unwrap().clone(),
            )
            .add_sampler(sampler)
            .unwrap()
            .build()
            .unwrap(),
        );
        let egui_texture = ImmutableImage::from_iter(
            [0u8].iter().copied(),
            Dimensions::Dim2d {
                width: 1,
                height: 1,
            },
            MipmapsCount::One,
            vulkano::format::Format::R8Unorm,
            queue.clone(),
        )
        .unwrap();
        request_sender
            .send((TextureId::Egui, egui_texture.1))
            .unwrap();
        let egui_texture_descriptor_set =
            Self::create_descriptor_set_from_view(pipeline.clone(), egui_texture.0);
        Self {
            pipeline,
            egui_texture_descriptor_set,
            egui_texture_version: None,
            user_textures: vec![],
            device,
            queue,
            frame_buffers: vec![],
            vertex_buffer_pool,
            index_buffer_pool,
            image_staging_buffer,
            descriptor_set_0,
            dynamic: Default::default(),

            image_transfer_request_list: vec![],
            request_sender,
            done_notifier,
        }
    }

    /// you must call when SwapChain or render target resize or before first create_command_buffer call
    pub fn create_frame_buffers<Wnd: Send + Sync + 'static>(
        &mut self,
        image_views: &[Arc<SwapchainImage<Wnd>>],
    ) {
        self.frame_buffers = image_views
            .iter()
            .map(|image| {
                Arc::new(
                    Framebuffer::start(self.pipeline.clone())
                        .add(image.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                ) as Arc<dyn FramebufferAbstract + Send + Sync>
            })
            .collect::<Vec<_>>();
    }
    /// execute command and present to screen
    pub fn present_to_screen<Wnd>(
        &self,
        command: AutoCommandBuffer<StandardCommandPoolAlloc>,
        acquire_future: SwapchainAcquireFuture<Wnd>,
    ) {
        let swap_chain = acquire_future.swapchain().clone();
        let image_id = acquire_future.image_id();

        if let Ok(ok) = acquire_future
            .then_execute(self.queue.clone(), command)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), swap_chain, image_id)
            .then_signal_fence_and_flush()
        {
            if ok.wait(None).is_ok() {};
        }
    }

    /// translate egui rendering request to vulkano AutoCommandBuffer
    ///
    /// color attachment is render target.
    ///
    /// If color_attachment is none then render to self framebuffer and image num must Some
    ///
    /// If color attachment is some then render to any image e.g. AttachmentImage StorageImage
    pub fn create_command_buffer(
        &mut self,
        color_attachment: Option<Arc<dyn ImageViewAccess + Send + Sync>>,
        image_num: Option<usize>,
        paint_jobs: &[ClippedMesh],
        screen_descriptor: &ScreenDescriptor,
    ) -> AutoCommandBuffer<StandardCommandPoolAlloc> {
        let mut exact_sized_iter_vec_vertices = ExactSizedIterVec::new();
        let mut exact_sized_iter_vec_indices = ExactSizedIterVec::new();
        let framebuffer = if let Some(color_attachment) = color_attachment {
            Arc::new(
                vulkano::framebuffer::Framebuffer::start(self.pipeline.render_pass().clone())
                    .add(color_attachment)
                    .unwrap()
                    .build()
                    .expect("failed to create frame buffer"),
            )
        } else {
            self.frame_buffers[image_num.unwrap()].clone()
        };
        let logical = screen_descriptor.logical_size();

        let mut pass = vulkano::command_buffer::AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .expect("failed to create command buffer builder");
        pass.begin_render_pass(
            framebuffer,
            SubpassContents::Inline,
            vec![[0.0, 0.0, 0.0, 0.0].into()],
        )
        .unwrap();

        let physical_height = screen_descriptor.physical_height;
        let physical_width = screen_descriptor.physical_width;

        self.dynamic.viewports = Some(vec![Viewport {
            origin: [0.0; 2],
            dimensions: [physical_width as f32, physical_height as f32],
            depth_range: Default::default(),
        }]);

        let mut all_mesh_range = Vec::with_capacity(paint_jobs.len());
        let clip_rectangles = paint_jobs.iter().map(|x| x.0);
        let scissors = skip_by_clip(screen_descriptor, clip_rectangles);
        let mut indices_from = 0;
        let mut vertices_from = 0;
        #[cfg(debug_assertions)]
        println!("start frame");
        for ((_i, egui::ClippedMesh(_, mesh)), scissor) in
            paint_jobs.iter().enumerate().zip(scissors.clone())
        {
            #[cfg(debug_assertions)]
            println!(
                "mesh {} vertices {} indices {}",
                _i,
                mesh.vertices.len(),
                mesh.indices.len()
            );
            if mesh.vertices.is_empty() || mesh.indices.is_empty() | scissor.is_none() {
                all_mesh_range.push(None);
                #[cfg(debug_assertions)]
                println!("Detect glitch mesh");
                continue;
            }
            exact_sized_iter_vec_indices.push(mesh.indices.as_slice());
            #[allow(clippy::transmute_ptr_to_ptr)]
            exact_sized_iter_vec_vertices
                .push(unsafe { std::mem::transmute(mesh.vertices.as_slice()) });

            let indices_len = mesh.indices.len();
            let vertices_len = mesh.vertices.len();
            let indices_end = indices_from + indices_len;
            let vertices_end = vertices_from + vertices_len;
            all_mesh_range.push(Some((
                indices_from..indices_end,
                vertices_from..vertices_end,
            )));
            indices_from = indices_end;
            vertices_from = vertices_end;
        }
        // put data to buffer
        let index_buffer = self
            .index_buffer_pool
            .chunk(exact_sized_iter_vec_indices.copied())
            .unwrap();
        let vertex_buffer = self
            .vertex_buffer_pool
            .chunk(exact_sized_iter_vec_vertices.copied())
            .unwrap();
        #[cfg(debug_assertions)]
        println!("end frame");
        for ((egui::ClippedMesh(_, mesh), range), scissor) in
            paint_jobs.iter().zip(all_mesh_range.iter()).zip(scissors)
        {
            if let Some(range) = range {
                let index_range = range.0.clone();
                let vertex_range = range.1.clone();
                //mesh to index and vertex buffer

                let texture_id = mesh.texture_id;
                let texture_desc_set = self.get_descriptor_set(texture_id);
                //emit valid texturing
                if scissor.is_some() {
                    let index_buffer = BufferSlice::from_typed_buffer_access(index_buffer.clone())
                        .slice(index_range)
                        .unwrap();
                    let vertex_buffer =
                        BufferSlice::from_typed_buffer_access(vertex_buffer.clone())
                            .slice(vertex_range)
                            .unwrap();
                    let marker = if texture_id == TextureId::Egui {
                        IS_EGUI
                    } else {
                        IS_USER
                    };
                    let pc = {
                        PushConstants {
                            u_screen_size: [logical.0 as f32, logical.1 as f32],
                            marker,
                        }
                    };
                    {
                        self.dynamic.scissors = Some(vec![scissor.unwrap()]);
                        pass.draw_indexed(
                            self.pipeline.clone(),
                            &self.dynamic,
                            vertex_buffer,
                            index_buffer,
                            (self.descriptor_set_0.clone(), texture_desc_set),
                            pc,
                            vec![],
                        )
                        .unwrap();
                    }
                }
            }
        }
        pass.end_render_pass().unwrap();
        pass.build().unwrap()
    }
    /// Update egui system texture
    /// You must call before every  create_command_buffer
    pub fn request_upload_egui_texture(&mut self, texture: &Texture) {
        //no change
        if self.egui_texture_version == Some(texture.version) {
            return;
        }
        let format = vulkano::format::Format::R8Unorm;
        let staging_sub_buffer = self
            .image_staging_buffer
            .chunk(texture.pixels.iter().copied())
            .unwrap();
        let image = vulkano::image::ImmutableImage::from_buffer(
            staging_sub_buffer,
            Dimensions::Dim2d {
                width: texture.width as u32,
                height: texture.height as u32,
            },
            MipmapsCount::One,
            format,
            self.queue.clone(),
        )
        .unwrap();
        self.request_sender
            .send((TextureId::Egui, image.1))
            .expect("failed to send system texture upload request");
        let image_view = image.0 as Arc<dyn ImageViewAccess + Sync + Send>;
        let pipeline = self.pipeline.clone();
        self.egui_texture_descriptor_set =
            Self::create_descriptor_set_from_view(pipeline, image_view);
        self.egui_texture_version = Some(texture.version);
    }

    fn alloc_user_texture(&mut self) -> TextureId {
        for (i, tex) in self.user_textures.iter_mut().enumerate() {
            if tex.is_none() {
                return TextureId::User(i as u64);
            }
        }
        let id = TextureId::User(self.user_textures.len() as u64);
        self.user_textures.push(None);
        id
    }
    fn free_user_texture(&mut self, id: TextureId) {
        if let TextureId::User(id) = id {
            #[cfg(debug_assertions)]
            println!("free {}", id);
            self.user_textures
                .get_mut(id as usize)
                .and_then(|option: &mut Option<UserTexture>| option.take());
        }
    }
    /// waiting image upload done.
    ///
    /// usually you don't need to call .
    ///
    /// this cause blocking but ensure no image glitch.
    ///
    /// you must call before  create command buffer
    pub fn wait_texture_upload(&mut self) {
        //wait
        // check request is not empty
        if !self.image_transfer_request_list.is_empty() {
            for done in self.done_notifier.iter() {
                #[cfg(debug_assertions)]
                println!("image upload finished : {:?}", done);
                for (index, request) in self.image_transfer_request_list.iter().enumerate() {
                    if *request == done {
                        self.image_transfer_request_list.remove(index);
                        break;
                    }
                }
                if self.image_transfer_request_list.is_empty() {
                    break;
                }
            }
        }
    }
    fn create_descriptor_set_from_view(
        pipeline: Arc<Pipeline>,
        image_view: Arc<dyn ImageViewAccess + Sync + Send>,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        Arc::new(
            PersistentDescriptorSet::start(
                pipeline.layout().descriptor_set_layout(1).unwrap().clone(),
            )
            .add_image(image_view)
            .unwrap()
            .build()
            .unwrap(),
        )
    }
    fn get_descriptor_set(&self, texture_id: TextureId) -> Arc<dyn DescriptorSet + Send + Sync> {
        if let egui::TextureId::User(id) = texture_id {
            self.user_textures
                .get(id as usize)
                .unwrap()
                .as_ref()
                .unwrap()
                .descriptor_set
                .clone()
        } else {
            self.egui_texture_descriptor_set.clone()
        }
    }
    pub fn set_user_texture(
        &mut self,
        id: TextureId,
        size: (usize, usize),
        srgba_pixels: &[Color32],
    ) {
        #[cfg(debug_assertions)]
        println!("new texture arrived {:?}", id);
        if let TextureId::User(id) = id {
            // test Texture slot allocated
            if let Some(slot) = self.user_textures.get_mut(id as usize) {
                // cast slice
                //
                let pixels = unsafe {
                    std::slice::from_raw_parts(
                        srgba_pixels.as_ptr() as *const u8,
                        srgba_pixels.len() * 4,
                    )
                };
                let sub_buffer = self
                    .image_staging_buffer
                    .chunk(pixels.iter().copied())
                    .unwrap();
                let (image, future) = vulkano::image::ImmutableImage::from_buffer(
                    sub_buffer,
                    Dimensions::Dim2d {
                        width: size.0 as u32,
                        height: size.1 as u32,
                    },
                    MipmapsCount::One,
                    R8G8B8A8Srgb,
                    self.queue.clone(),
                )
                .unwrap();
                self.request_sender
                    .send((TextureId::User(id), future))
                    .expect("faild to send image upload request");
                self.image_transfer_request_list
                    .push(TextureId::User(id as u64));
                let image_view = image as Arc<dyn ImageViewAccess + Sync + Send>;
                let desc = Self::create_descriptor_set_from_view(self.pipeline.clone(), image_view);
                *slot = Some(UserTexture {
                    descriptor_set: desc,
                });
            }
        }
    }
    /// register vulkano image view as egui texture
    ///
    /// Usable for render to image rectangle

    pub fn register_vulkano_texture(
        &mut self,
        image_view: Arc<dyn ImageViewAccess + Sync + Send>,
    ) -> TextureId {
        let id = self.alloc_user_texture();
        if let egui::TextureId::User(id) = id {
            let pipeline = self.pipeline.clone();
            if let Some(slot) = self.user_textures.get_mut(id as usize) {
                *slot = Some(UserTexture {
                    descriptor_set: Self::create_descriptor_set_from_view(pipeline, image_view),
                });
            }
        }
        id
    }
}
impl epi::TextureAllocator for EguiVulkanoRenderPass {
    fn alloc_srgba_premultiplied(
        &mut self,
        size: (usize, usize),
        srgba_pixels: &[Color32],
    ) -> TextureId {
        let id = self.alloc_user_texture();
        self.set_user_texture(id, size, srgba_pixels);
        id
    }
    fn free(&mut self, id: TextureId) {
        self.free_user_texture(id)
    }
}
fn skip_by_clip(
    screen_desc: &ScreenDescriptor,
    clip_rectangles: impl Iterator<Item = egui::Rect> + Clone,
) -> impl Iterator<Item = Option<Scissor>> + Clone {
    let scale_factor = screen_desc.scale_factor;
    let physical_width = screen_desc.physical_width;
    let physical_height = screen_desc.physical_height;
    clip_rectangles.map(move |clip_rect| {
        // Transform clip rect to physical pixels.
        let clip_min_x = scale_factor * clip_rect.min.x;
        let clip_min_y = scale_factor * clip_rect.min.y;
        let clip_max_x = scale_factor * clip_rect.max.x;
        let clip_max_y = scale_factor * clip_rect.max.y;

        // Make sure clip rect can fit within an `u32`.
        let clip_min_x = clip_min_x.clamp(0.0, physical_width as f32);
        let clip_min_y = clip_min_y.clamp(0.0, physical_height as f32);
        let clip_max_x = clip_max_x.clamp(clip_min_x, physical_width as f32);
        let clip_max_y = clip_max_y.clamp(clip_min_y, physical_height as f32);

        let clip_min_x = clip_min_x.round() as u32;
        let clip_min_y = clip_min_y.round() as u32;
        let clip_max_x = clip_max_x.round() as u32;
        let clip_max_y = clip_max_y.round() as u32;

        let width = (clip_max_x - clip_min_x).max(1);
        let height = (clip_max_y - clip_min_y).max(1);
        // clip scissor rectangle to target size
        let x = clip_min_x.min(physical_width);
        let y = clip_min_y.min(physical_height);
        let width = width.min(physical_width - x);
        let height = height.min(physical_height - y);
        // skip rendering with zero-sized clip areas
        if width == 0 || height == 0 {
            None
        } else {
            Some(Scissor {
                origin: [x as i32, y as i32],
                dimensions: [width, height],
            })
        }
    })
}

struct UserTexture {
    descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
}
/// four thread for rendering
/// thread 0 | thread 1| thread 2 | thread 3|
/// management | image upload| command build |present+rendering |
///

pub struct RenderingDispatcher<Window: 'static + Send + Sync> {
    inner: Arc<Mutex<EguiVulkanoRenderPass>>,
    swap_chain: Arc<Swapchain<Window>>,
    build_request_sender: Sender<BuildRequest>,
    render_request_sender: Sender<RenderRequest<Window>>,
    command_buffer_receiver: Receiver<CommandBuffer>,
}
type RenderRequest<Window> = (AutoCommandBuffer, SwapchainAcquireFuture<Window>);
type BuildRequest = (usize, Vec<ClippedMesh>, ScreenDescriptor);
type CommandBuffer = AutoCommandBuffer;
impl<Window: Send + Sync> RenderingDispatcher<Window> {
    pub fn build(
        device: Arc<Device>,
        queue: Arc<Queue>,
        swap_chain_images: &[Arc<SwapchainImage<Window>>],
    ) -> Self {
        let swap_chain = swap_chain_images[0].swapchain().clone();
        let format = swap_chain.format();
        let inner = Arc::new(Mutex::new(EguiVulkanoRenderPass::new(
            device,
            queue.clone(),
            format,
        )));
        inner
            .lock()
            .unwrap()
            .create_frame_buffers(&swap_chain_images);
        //rendering request channel
        let (render_request_sender, render_request_receiver): (
            Sender<RenderRequest<Window>>,
            Receiver<RenderRequest<Window>>,
        ) = std::sync::mpsc::channel();

        let (build_request_sender, build_request_receiver): (
            Sender<BuildRequest>,
            Receiver<BuildRequest>,
        ) = std::sync::mpsc::channel();
        let (command_buffer_sender, command_buffer_receiver) = std::sync::mpsc::channel();
        // spawn threads

        //rendering and present thread
        std::thread::spawn(move || {
            for render_request in render_request_receiver.iter() {
                //   inner.present_to_screen(command_buffer,future);

                let acquire_future = render_request.1;
                let command = render_request.0;
                let swap_chain = acquire_future.swapchain().clone();
                let image_id = acquire_future.image_id();

                if let Ok(ok) = acquire_future
                    .then_execute(queue.clone(), command)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swap_chain, image_id)
                    .then_signal_fence_and_flush()
                {
                    if ok.wait(None).is_ok() {};
                }
            }
        });
        // command build thread
        let clone_inner = inner.clone();
        std::thread::spawn(move || {
            for (x, paint_job, screen_descriptor) in build_request_receiver.iter() {
                let command = clone_inner.lock().unwrap().create_command_buffer(
                    None,
                    Some(x),
                    &paint_job,
                    &screen_descriptor,
                );
                command_buffer_sender.send(command).unwrap();
            }
        });
        Self {
            inner,
            swap_chain,
            build_request_sender,
            render_request_sender,
            command_buffer_receiver,
        }
    }
    pub fn create_frame_buffers(&self, images: &[Arc<SwapchainImage<Window>>]) {
        self.inner.lock().unwrap().create_frame_buffers(images);
    }
    pub fn render(&mut self, job: Vec<ClippedMesh>, screen_desc: ScreenDescriptor) -> bool {
        //get new frame
        let (image_num, _sub_optimal, future) =
            match vulkano::swapchain::acquire_next_image(self.swap_chain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    //recreate_swapchain = true;
                    println!("out of date");
                    return true;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };
        println!("acquire next image");
        //add new rendering request and pull command

        self.build_request_sender
            .send((image_num, job, screen_desc))
            .unwrap();
        println!("command build request sent ");
        let command_buffer = self.command_buffer_receiver.recv().unwrap();
        self.render_request_sender
            .send((command_buffer, future))
            .unwrap();
        false
    }
    pub fn upload_egui_texture(&self, texture: &epi::egui::Texture) {
        self.inner
            .lock()
            .unwrap()
            .request_upload_egui_texture(texture)
    }
}
impl<W: Send + Sync> epi::TextureAllocator for RenderingDispatcher<W> {
    fn alloc_srgba_premultiplied(
        &mut self,
        size: (usize, usize),
        srgba_pixels: &[Color32],
    ) -> TextureId {
        self.inner
            .lock()
            .unwrap()
            .alloc_srgba_premultiplied(size, srgba_pixels)
    }

    fn free(&mut self, id: TextureId) {
        self.inner.lock().unwrap().free(id)
    }
}
