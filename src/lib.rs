//! [egui](https://docs.rs/egui) rendering integration for vulkano
//!
//! This crate have some features
//! * background user texture upload
//! * map vulkano texture to user texture
//! * lower memory usage
extern crate vulkano;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use epi::egui;
use epi::egui::{ClippedMesh, Color32, Texture, TextureId};
use iter_vec::ExactSizedIterVec;
use vulkano::buffer::{BufferSlice, BufferUsage, CpuBufferPool};
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::command_buffer::{
    CommandBufferExecFuture, CommandBufferUsage, DynamicState, PrimaryAutoCommandBuffer,
    SubpassContents,
};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::DescriptorSet;
use vulkano::device::{Device, Queue};
use vulkano::image::view::{ImageView, ImageViewAbstract, ImageViewCreationError};
use vulkano::image::{
    AttachmentImage, ImageCreationError, ImageDimensions, ImageUsage, MipmapsCount, SwapchainImage,
};
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::{Scissor, Viewport};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::render_pass::{Framebuffer, FramebufferAbstract};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::SwapchainAcquireFuture;
use vulkano::sync::{GpuFuture, NowFuture};

use crate::shader::create_pipeline;

mod render_pass;

mod shader;
/*
same layout as EGUI
memory layout is specified by Vertex trait
so we can cast from Vertex and send it to GPU directly
*/
#[repr(C)]
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
/// same as [egui_wgpu_backend::ScreenDescriptor](https://docs.rs/egui_wgpu_backend/0.8.0/egui_wgpu_backend/struct.ScreenDescriptor.html)
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

type Pipeline = GraphicsPipeline<SingleBufferDefinition<EguiVulkanoVertex>>;

/// egui rendering command builder
pub struct EguiVulkanoRenderPass {
    pipeline: Arc<Pipeline>,
    egui_texture_version: Option<u64>,
    egui_textures: BTreeMap<Option<u64>, Option<TextureDescriptor>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    frame_buffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    vertex_buffer_pool: CpuBufferPool<EguiVulkanoVertex>,
    index_buffer_pool: CpuBufferPool<u32>,
    image_staging_buffer: CpuBufferPool<u8>,
    sampler_desc_set: Arc<dyn DescriptorSet + Send + Sync>,
    //set=0 binding=0
    dynamic: DynamicState,
    image_transfer_requests: BTreeSet<Option<u64>>,
    request_sender: Sender<Work>,
    done_notifier: Receiver<Option<u64>>,
}

fn texture_id_as_option_u64(x: TextureId) -> Option<u64> {
    match x {
        TextureId::Egui => None,
        TextureId::User(id) => Some(id),
    }
}

type Work = (
    Option<u64>,
    CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>,
);

fn spawn_image_upload_thread() -> (Sender<Work>, Receiver<Option<u64>>) {
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
/*
We use Option<u64> as TextureId
TextureId does not implement Ord trait
TextureId => Option<u64>
Egui => None
User(id) => Some(id)
*/
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
        let sampler_desc_set = Arc::new(
            PersistentDescriptorSet::start(
                pipeline.layout().descriptor_set_layout(0).unwrap().clone(),
            )
            .add_sampler(sampler)
            .unwrap()
            .build()
            .unwrap(),
        ) as Arc<dyn DescriptorSet + Send + Sync>;

        Self {
            pipeline,
            egui_texture_version: None,
            egui_textures: BTreeMap::new(),
            device,
            queue,
            frame_buffers: vec![],
            vertex_buffer_pool,
            index_buffer_pool,
            image_staging_buffer,
            sampler_desc_set,
            dynamic: Default::default(),
            image_transfer_requests: BTreeSet::new(),
            request_sender,
            done_notifier,
        }
    }

    /// you must call when SwapChain  resize or before first create_command_buffer call
    pub fn create_frame_buffers<Wnd: Send + Sync + 'static>(
        &mut self,
        image_views: &[Arc<SwapchainImage<Wnd>>],
    ) {
        self.frame_buffers = image_views
            .iter()
            .map(|image| {
                let view = ImageView::new(image.clone()).unwrap();
                Arc::new(
                    Framebuffer::start(self.pipeline.clone().render_pass().clone())
                        .add(view)
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
        command: PrimaryAutoCommandBuffer<StandardCommandPoolAlloc>,
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
    pub fn create_command_buffer(
        &mut self,
        render_target: RenderTarget,
        paint_jobs: &[ClippedMesh],
        screen_descriptor: &ScreenDescriptor,
    ) -> PrimaryAutoCommandBuffer<StandardCommandPoolAlloc> {
        let mut exact_sized_iter_vec_vertices = ExactSizedIterVec::new();
        let mut exact_sized_iter_vec_indices = ExactSizedIterVec::new();
        let framebuffer = match render_target {
            RenderTarget::ColorAttachment(color_attachment) => Arc::new(
                vulkano::render_pass::Framebuffer::start(self.pipeline.render_pass().clone())
                    .add(color_attachment)
                    .unwrap()
                    .build()
                    .expect("[BackEnd] failed to create frame buffer"),
            ),
            RenderTarget::FrameBufferIndex(id) => self.frame_buffers[id].clone(),
        };

        let logical = screen_descriptor.logical_size();

        let mut pass = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("[BackEnd] failed to create command buffer builder");
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
            depth_range: 0.0..1.0,
        }]);
        /*
        View of Vertex and Index buffer
        */
        let mut all_mesh_range = Vec::with_capacity(paint_jobs.len());
        let clip_rectangles = paint_jobs.iter().map(|x| x.0);
        let scissors = skip_by_clip(screen_descriptor, clip_rectangles);
        let mut indices_from = 0;
        let mut vertices_from = 0;
        #[cfg(feature = "backend_debug")]
        println!("[BackEnd] start frame");
        /*
        upload index buffer and vertex buffer and remove invalid mesh
        we can parallel processing for this work but egui tessellation is slow enough when Fractal clock with 14 depth
        almost 70% of cpu time is spent for tessellation
        */
        for ((_i, egui::ClippedMesh(_, mesh)), scissor) in
            paint_jobs.iter().enumerate().zip(scissors.clone())
        {
            #[cfg(feature = "backend_debug")]
            println!(
                "[BackEnd] mesh {} vertices {} indices {} texture_id {:?}",
                _i,
                mesh.vertices.len(),
                mesh.indices.len(),
                mesh.texture_id
            );
            /*
            remove index buffer or vertex buffer is empty or clip rectangle is 0 sized area
            */
            if mesh.vertices.is_empty() || mesh.indices.is_empty() | scissor.is_none() {
                #[cfg(feature = "backend_debug")]
                println!("[BackEnd] Detect glitch mesh");
                continue;
            }
            exact_sized_iter_vec_indices.push(mesh.indices.as_slice());
            /*
            SAFETY: GPU send only cast
            */
            #[allow(clippy::transmute_ptr_to_ptr)]
            exact_sized_iter_vec_vertices
                .push(unsafe { std::mem::transmute(mesh.vertices.as_slice()) });

            let indices_len = mesh.indices.len();
            let vertices_len = mesh.vertices.len();
            let indices_end = indices_from + indices_len;
            let vertices_end = vertices_from + vertices_len;
            all_mesh_range.push((
                mesh.texture_id,
                (indices_from..indices_end, vertices_from..vertices_end),
            ));
            indices_from = indices_end;
            vertices_from = vertices_end;
        }
        // put data to buffer
        /*
        generalized for iterator so this is bottle neck
        We may need to parallel uploading.
        */
        #[cfg(feature = "backend_debug")]
        let instant = std::time::Instant::now();

        let index_buffer = self
            .index_buffer_pool
            .chunk(exact_sized_iter_vec_indices.copied())
            .unwrap();
        let vertex_buffer = self
            .vertex_buffer_pool
            .chunk(exact_sized_iter_vec_vertices.copied())
            .unwrap();
        #[cfg(feature = "backend_debug")]
        println!("[BackEnd] upload finished in {:?}", instant.elapsed());
        #[cfg(feature = "backend_debug")]
        println!("[BackEnd] end frame");
        for ((texture_id, range), scissor) in all_mesh_range.iter().zip(scissors) {
            let index_range = range.0.clone();
            let vertex_range = range.1.clone();
            //mesh to index and vertex buffer

            let texture_desc_set = self.get_descriptor_set(*texture_id);
            //emit valid texturing
            if scissor.is_some() {
                let index_buffer = BufferSlice::from_typed_buffer_access(index_buffer.clone())
                    .slice(index_range)
                    .unwrap();
                let vertex_buffer = BufferSlice::from_typed_buffer_access(vertex_buffer.clone())
                    .slice(vertex_range)
                    .unwrap();
                let marker = if *texture_id == TextureId::Egui {
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
                        (self.sampler_desc_set.clone(), texture_desc_set),
                        pc,
                        vec![],
                    )
                    .unwrap();
                }
            }
        }
        pass.end_render_pass().unwrap();
        pass.build().unwrap()
    }
    /// Update egui system texture
    /// You must call before every  create_command_buffer when drawing content changed
    pub fn request_upload_egui_texture(&mut self, texture: &Texture) {
        //no change
        if self.egui_texture_version == Some(texture.version) {
            return;
        }
        let staging_sub_buffer = self
            .image_staging_buffer
            .chunk(texture.pixels.iter().copied())
            .unwrap();
        let image = vulkano::image::ImmutableImage::from_buffer(
            staging_sub_buffer,
            ImageDimensions::Dim2d {
                width: texture.width as u32,
                height: texture.height as u32,
                array_layers: 1,
            },
            MipmapsCount::One,
            vulkano::format::Format::R8Unorm,
            self.queue.clone(),
        )
        .unwrap();
        self.request_sender
            .send((None, image.1))
            .expect("[BackEnd] failed to send system texture upload request");
        let image_view = ImageView::new(image.0).unwrap();
        let image_view = image_view as Arc<dyn ImageViewAbstract + Sync + Send>;
        let pipeline = self.pipeline.clone();
        let texture_desc = Self::create_descriptor_set_from_view(pipeline, image_view);
        self.egui_textures
            .insert(None, Some(TextureDescriptor(texture_desc)));
        self.egui_texture_version = Some(texture.version);
    }

    fn alloc_user_texture(&mut self) -> TextureId {
        /*
        if we find empty slot
        */
        for (i, tex) in self.egui_textures.iter() {
            if tex.is_none() && i.is_some() {
                return TextureId::User(i.unwrap());
            }
        }
        /*
                Compatibility for glium backend .
            Because glium's TextureId start from 0.
        */
        let id = if self.egui_textures.contains_key(&None) {
            self.egui_textures.len() - 1
        } else {
            self.egui_textures.len()
        };
        self.egui_textures.insert(Some(id as u64), None);
        TextureId::User(id as u64)
    }
    fn free_user_texture(&mut self, id: TextureId) {
        let t_id = texture_id_as_option_u64(id);
        self.egui_textures
            .get_mut(&t_id)
            .and_then(|option| option.take());
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
        if !self.image_transfer_requests.is_empty() {
            for done in self.done_notifier.iter() {
                #[cfg(feature = "backend_debug")]
                println!("[BackEnd] image upload finished : {:?}", done);
                self.image_transfer_requests.remove(&done);
                if self.image_transfer_requests.is_empty() {
                    break;
                }
            }
        }
    }
    fn create_descriptor_set_from_view(
        pipeline: Arc<Pipeline>,
        image_view: Arc<dyn ImageViewAbstract + Sync + Send>,
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
        let t_id = texture_id_as_option_u64(texture_id);
        self.egui_textures
            .get(&t_id)
            .unwrap()
            .as_ref()
            .unwrap()
            .0
            .clone()
    }
    pub fn set_user_texture(
        &mut self,
        id: TextureId,
        size: (usize, usize),
        srgba_pixels: &[Color32],
    ) {
        #[cfg(feature = "backend_debug")]
        println!("new texture arrived {:?}", id);
        let t_id = texture_id_as_option_u64(id);
        let image_staging_buffer = self.image_staging_buffer.clone();
        let pipeline = self.pipeline.clone();
        let queue = self.queue.clone();
        let sender = self.request_sender.clone();
        let is_executed =
            self.egui_textures
                .get_mut(&t_id)
                .map(|slot: &mut Option<TextureDescriptor>| {
                    // SAFETY: GPU send only cast
                    let pixels = unsafe {
                        std::slice::from_raw_parts(
                            srgba_pixels.as_ptr() as *const u8,
                            srgba_pixels.len() * 4,
                        )
                    };
                    let sub_buffer = image_staging_buffer.chunk(pixels.iter().copied()).unwrap();
                    let (image, future) = vulkano::image::ImmutableImage::from_buffer(
                        sub_buffer,
                        ImageDimensions::Dim2d {
                            width: size.0 as u32,
                            height: size.1 as u32,
                            array_layers: 1,
                        },
                        MipmapsCount::One,
                        vulkano::format::Format::R8G8B8A8Srgb,
                        queue,
                    )
                    .unwrap();
                    sender
                        .send((t_id, future))
                        .expect("failed to send image copy request");

                    let image_view = ImageView::new(image).unwrap();
                    let image_view = image_view as Arc<dyn ImageViewAbstract + Sync + Send>;
                    let desc = Self::create_descriptor_set_from_view(pipeline, image_view);
                    slot.replace(TextureDescriptor(desc));
                });
        if is_executed.is_some() {
            self.image_transfer_requests.insert(t_id);
        }
    }
    /// register vulkano image view as egui texture
    ///
    /// Usable for render to image rectangle

    pub fn register_vulkano_image_view(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Sync + Send>,
    ) -> TextureId {
        let id = self.alloc_user_texture();
        let t_id = texture_id_as_option_u64(id);
        if let Some(slot) = self.egui_textures.get_mut(&t_id) {
            slot.replace(TextureDescriptor(Self::create_descriptor_set_from_view(
                self.pipeline.clone(),
                image_view,
            )));
        }
        id
    }
    /// init render area with given dimensions.
    ///
    /// image format is R8G8B8A8Srgb.
    ///
    /// enabled image usage is below.
    /// * sampled
    /// * color_attachment
    /// * transfer_destination
    ///
    ///  this is shortcut function for register_vulkano_image_view
    ///
    /// example usage
    /// * model viewer
    /// * video playback
    pub fn init_vulkano_image_with_dimensions(
        &mut self,
        dimensions: [u32; 2],
    ) -> Result<(TextureId, Arc<AttachmentImage>), InitRenderAreaError> {
        let usage = ImageUsage {
            transfer_source: false,
            transfer_destination: true,
            sampled: true,
            storage: false,
            color_attachment: true,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        };
        self.init_vulkano_image_with_parameters(
            dimensions,
            usage,
            vulkano::format::Format::R8G8B8A8Srgb,
        )
    }
    pub fn init_vulkano_image_with_parameters(
        &mut self,
        dimensions: [u32; 2],
        usage: ImageUsage,
        format: vulkano::format::Format,
    ) -> Result<(TextureId, Arc<AttachmentImage>), InitRenderAreaError> {
        let pipeline = self.pipeline.clone();
        match AttachmentImage::with_usage(self.device.clone(), dimensions, format, usage) {
            Ok(image) => {
                let texture_id = self.alloc_user_texture();
                let t_id = texture_id_as_option_u64(texture_id);

                self.egui_textures
                    .get_mut(&t_id)
                    .map(|slot| {
                        return match ImageView::new(image.clone()) {
                            Ok(image_view) => {
                                slot.replace(TextureDescriptor(
                                    Self::create_descriptor_set_from_view(pipeline, image_view),
                                ));
                                Ok((texture_id, image))
                            }
                            Err(err) => Err(InitRenderAreaError::ImageViewCreationError(err)),
                        };
                    })
                    .unwrap()
            }
            Err(err) => Err(InitRenderAreaError::ImageCreationError(err)),
        }
    }
    ///  recreate vulkano texture.
    ///
    ///  usable for render target resize.
    ///
    ///  this function create descriptor set for image, so you do not have to call [register_vulkano_image_view]
    pub fn recreate_vulkano_texture_with_dimensions(
        &mut self,
        texture_id: TextureId,
        dimensions: [u32; 2],
    ) -> Result<Arc<AttachmentImage>, RecreateErrors> {
        let descriptor_set = self.get_descriptor_set(texture_id);
        let image_access = descriptor_set.image(0).unwrap().0.image();
        let format = image_access.format();
        let usage = image_access.inner().image.usage();
        let t_id = texture_id_as_option_u64(texture_id);
        if t_id.is_none() {
            Err(RecreateErrors::TextureIdIsEgui)
        } else {
            // create image with parameter
            match AttachmentImage::with_usage(self.device.clone(), dimensions, format, usage) {
                Ok(image) => {
                    match ImageView::new(image.clone()) {
                        Ok(image_view) => {
                            let descriptor = Self::create_descriptor_set_from_view(
                                self.pipeline.clone(),
                                image_view,
                            );
                            let texture_desc = TextureDescriptor(descriptor);
                            // register new descriptor set
                            if let Some(slot) = self.egui_textures.get_mut(&t_id) {
                                slot.replace(texture_desc);
                            }
                            Ok(image)
                        }
                        Err(err) => Err(RecreateErrors::ImageViewCreationError(err)),
                    }
                }
                Err(err) => Err(RecreateErrors::ImageCreationError(err)),
            }
        }
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
/*
remove non visible rectangle
*/
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

#[derive(Debug, Clone)]
pub enum RecreateErrors {
    TextureIdIsEgui,
    TextureIdNotFound(TextureId),
    ImageViewCreationError(ImageViewCreationError),
    ImageCreationError(ImageCreationError),
}

#[derive(Debug, Clone)]
pub enum InitRenderAreaError {
    ImageViewCreationError(ImageViewCreationError),
    ImageCreationError(ImageCreationError),
}

pub enum RenderTarget {
    /// render to texture or any other use
    ColorAttachment(Arc<dyn ImageViewAbstract + Send + Sync>),
    /// render to own framebuffer
    FrameBufferIndex(usize),
}

struct TextureDescriptor(Arc<dyn DescriptorSet + Send + Sync>);
