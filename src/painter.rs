use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use crate::shader::create_pipeline;
use epi::egui;
use epi::egui::{ClippedMesh, Color32, Texture, TextureId};

use crate::{RenderTarget, ScreenDescriptor};
use log::debug;
use threadpool::ThreadPool;
use vulkano::buffer::cpu_pool::CpuBufferPoolChunk;
use vulkano::buffer::{BufferUsage, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::{Device, Queue};
use vulkano::image::view::{ImageView, ImageViewAbstract, ImageViewCreationError};
use vulkano::image::{
    AttachmentImage, ImageAccess, ImageCreationError, ImageDimensions, ImageUsage, MipmapsCount,
};
use vulkano::memory::pool::StdMemoryPool;
use vulkano::pipeline::vertex::{VertexMemberInfo, VertexMemberTy};
use vulkano::pipeline::viewport::{Scissor, Viewport};
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferAbstract};
use vulkano::swapchain::SwapchainAcquireFuture;
use vulkano::sync::GpuFuture;
#[repr(C)]
struct IsEguiTextureMarker(u32);

const IS_EGUI: IsEguiTextureMarker = IsEguiTextureMarker(1);
const IS_USER: IsEguiTextureMarker = IsEguiTextureMarker(0);

#[repr(C, packed)]
pub(crate) struct PushConstants {
    u_screen_size: [f32; 2],
    marker: IsEguiTextureMarker,
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct WrappedEguiVertex(epi::egui::epaint::Vertex);
unsafe impl vulkano::pipeline::vertex::Vertex for WrappedEguiVertex {
    fn member(name: &str) -> Option<VertexMemberInfo> {
        let dummy = WrappedEguiVertex(epi::egui::epaint::Vertex::default());
        match name {
            "a_pos" => Some(VertexMemberInfo {
                offset: ((&dummy.0.pos) as *const _ as usize) - ((&dummy) as *const _ as usize),
                ty: VertexMemberTy::F32,
                array_size: 2,
            }),
            "a_tex_coord" => Some(VertexMemberInfo {
                offset: ((&dummy.0.uv) as *const _ as usize) - ((&dummy) as *const _ as usize),
                ty: VertexMemberTy::F32,
                array_size: 2,
            }),
            "a_color" => Some(VertexMemberInfo {
                offset: ((&dummy.0.color) as *const _ as usize) - ((&dummy) as *const _ as usize),
                ty: VertexMemberTy::U32,
                array_size: 1,
            }),
            _ => None,
        }
    }
}

struct RenderResource {
    index_buffer: CpuBufferPoolChunk<u32, Arc<StdMemoryPool>>,
    vertex_buffer: CpuBufferPoolChunk<WrappedEguiVertex, Arc<StdMemoryPool>>,
    texture_id: TextureId,
    scissor: vulkano::pipeline::viewport::Scissor,
    job_id: usize,
}
/// egui rendering command builder
pub struct Painter {
    pipeline: Arc<GraphicsPipeline>,
    egui_texture_version: Option<u64>,
    egui_textures: BTreeMap<Option<u64>, Option<TextureDescriptor>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    pub(crate) frame_buffers: Vec<Arc<dyn FramebufferAbstract>>,
    vertex_buffer_pool: CpuBufferPool<WrappedEguiVertex>,
    index_buffer_pool: CpuBufferPool<u32>,
    image_staging_buffer: CpuBufferPool<u8>,
    thread_pool: ThreadPool,
    render_resource_tx: Sender<RenderResource>,
    render_resource_rx: Receiver<RenderResource>,
}

fn texture_id_as_option_u64(x: TextureId) -> Option<u64> {
    match x {
        TextureId::Egui => None,
        TextureId::User(id) => Some(id),
    }
}
/*
We use Option<u64> as TextureId
TextureId does not implement Ord trait
TextureId => Option<u64>
Egui => None
User(id) => Some(id)
*/

impl Painter {
    ///create command builder
    ///
    /// if render target format incompatible with SwapChain format may cause color glitch
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_target_format: vulkano::format::Format,
    ) -> Self {
        debug!("starting painter init");
        let pipeline = create_pipeline(device.clone(), render_target_format);
        debug!("pipeline created");
        let thread_pool = ThreadPool::new(num_cpus::get());
        let vertex_buffer_pool = CpuBufferPool::vertex_buffer(device.clone());
        let index_buffer_pool = CpuBufferPool::new(device.clone(), BufferUsage::index_buffer());
        let image_staging_buffer =
            CpuBufferPool::new(device.clone(), BufferUsage::transfer_source());
        let (buffers_tx, buffers_rx) = std::sync::mpsc::channel();

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
            thread_pool,
            render_resource_tx: buffers_tx,
            render_resource_rx: buffers_rx,
        }
    }

    /// you must call when SwapChain  resize or before first create_command_buffer call
    pub fn create_frame_buffers<I: ImageAccess + 'static>(&mut self, swap_chain_images: &[Arc<I>]) {
        self.frame_buffers = swap_chain_images
            .iter()
            .map(|image| {
                let view = ImageView::new(image.clone()).unwrap();
                let fb = Framebuffer::start(self.pipeline.subpass().render_pass().clone())
                    .add(view)
                    .unwrap()
                    .build()
                    .unwrap();
                Arc::new(fb) as Arc<dyn FramebufferAbstract>
            })
            .collect();
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
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        render_target: RenderTarget,
        paint_jobs: Vec<ClippedMesh>,
        screen_descriptor: &ScreenDescriptor,
    ) {
        let framebuffer = match render_target {
            RenderTarget::ColorAttachment(color_attachment) => {
                let fb = Framebuffer::start(self.pipeline.subpass().render_pass().clone())
                    .add(color_attachment)
                    .unwrap();
                Arc::new(fb.build().expect("[BackEnd] failed to create frame buffer"))
            }
            RenderTarget::FrameBufferIndex(id) => self.frame_buffers[id].clone(),
        };
        let physical_height = screen_descriptor.physical_height;
        let physical_width = screen_descriptor.physical_width;
        let logical = screen_descriptor.logical_size();
        command_buffer_builder.set_viewport(
            0,
            vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [physical_width as f32, physical_height as f32],
                depth_range: 0.0..1.0,
            }],
        );
        command_buffer_builder.bind_pipeline_graphics(self.pipeline.clone());

        command_buffer_builder
            .begin_render_pass(framebuffer, SubpassContents::Inline, vec![[0.0; 4].into()])
            .unwrap();

        #[cfg(feature = "backend_debug")]
        println!("[BackEnd] start frame");

        #[cfg(feature = "backend_debug")]
        let instant = std::time::Instant::now();
        let mut job_count = 0;

        for egui::ClippedMesh(rect, mesh) in paint_jobs {
            let scissor = calc_scissor(rect, *screen_descriptor);
            if mesh.is_empty() || scissor.is_none() {
                #[cfg(feature = "backend_debug")]
                println!("[BackEnd] Detect glitch mesh");
                continue;
            }
            let index_buffer_pool = self.index_buffer_pool.clone();
            let vertex_buffer_pool = self.vertex_buffer_pool.clone();
            let tx = self.render_resource_tx.clone();
            /*Data uploading*/
            self.thread_pool.execute(move || {
                let texture_id = mesh.texture_id;
                #[cfg(feature = "backend_debug")]
                println!("[BackEnd] Fission");
                #[cfg(feature = "backend_debug")]
                println!(
                    "[BackEnd] mesh {}  vertices {} indices {} texture_id {:?}",
                    job_count,
                    mesh.vertices.len(),
                    mesh.indices.len(),
                    texture_id
                );
                let vertex_slice: &[WrappedEguiVertex] =
                    bytemuck::cast_slice(mesh.vertices.as_ref());
                let index = index_buffer_pool
                    .chunk(mesh.indices.iter().copied())
                    .unwrap();
                let vertex = vertex_buffer_pool
                    .chunk(vertex_slice.iter().copied())
                    .unwrap();
                tx.send(RenderResource {
                    index_buffer: index,
                    vertex_buffer: vertex,
                    texture_id,
                    scissor: scissor.unwrap(),
                    job_id: job_count,
                })
                .unwrap();
            });
            job_count += 1;
        }
        let draw = |render_resource: RenderResource| {
            let texture_id = render_resource.texture_id;
            let texture = self.get_descriptor_set(texture_id);
            let marker = match texture_id {
                TextureId::User(_) => IS_USER,
                _ => IS_EGUI,
            };

            let pc = PushConstants {
                u_screen_size: [logical.0 as f32, logical.1 as f32],
                marker,
            };
            let indices = render_resource.index_buffer.len() as u32;
            command_buffer_builder.push_constants(self.pipeline.layout().clone(), 0, pc);
            command_buffer_builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                texture,
            );
            command_buffer_builder.bind_vertex_buffers(0, render_resource.vertex_buffer);
            command_buffer_builder.bind_index_buffer(render_resource.index_buffer);
            command_buffer_builder.set_scissor(0, vec![render_resource.scissor]);
            command_buffer_builder
                .draw_indexed(indices, 1, 0, 0, 0)
                .unwrap();
        };

        {
            /*
                To avoid image flicker , We sort by job id
                We must take job_counts commands to avoid blocking
            */

            let mut jobs: Vec<RenderResource> =
                self.render_resource_rx.iter().take(job_count).collect();
            #[cfg(feature = "backend_debug")]
            println!("[BackEnd] upload finished in {:?}", instant.elapsed());
            #[cfg(feature = "backend_debug")]
            println!("Fusion {} meshes", job_count);
            jobs.sort_by_key(|render_resource| render_resource.job_id);
            let _: Vec<()> = jobs.into_iter().map(draw).collect();
        }
        #[cfg(feature = "backend_debug")]
        println!("[BackEnd] end frame");
        command_buffer_builder.end_render_pass().unwrap();
    }
    /// Update egui system texture
    /// You must call before every  create_command_buffer when drawing content changed
    pub fn request_upload_egui_texture(&mut self, texture: &Texture) {
        assert!(
            (texture.width * texture.height) >= texture.pixels.len(),
            "Invalid pixels in EGUI system texture given {} x {} actual {} pixels",
            texture.width,
            texture.height,
            texture.pixels.len()
        );
        self.set_texture(
            TextureId::Egui,
            &texture.pixels,
            (texture.width as u32, texture.height as u32),
            Some(texture.version),
        )
    }
    #[inline(always)]
    fn alloc_user_texture(&mut self) -> TextureId {
        //if we find empty slot
        //refuse egui system texture
        if let Some(empty_slot) = self
            .egui_textures
            .iter()
            .find(|entry| entry.0.is_some() && entry.1.is_none())
        {
            return TextureId::User(empty_slot.0.unwrap());
        };
        /*
                Compatibility for glium backend .
            Because glium's TextureId start from 0.
        */
        let id = if self.egui_textures.contains_key(&None) {
            self.egui_textures.len() - 1
        } else {
            self.egui_textures.len()
        };
        //allocate new empty slot
        self.egui_textures.insert(Some(id as u64), None);
        TextureId::User(id as u64)
    }
    #[inline(always)]
    fn free_user_texture(&mut self, id: TextureId) {
        let t_id = texture_id_as_option_u64(id);
        self.egui_textures
            .get_mut(&t_id)
            .and_then(|option| option.take());
    }
    #[inline(always)]
    fn create_descriptor_set_from_view(
        pipeline: Arc<GraphicsPipeline>,
        image_view: Arc<dyn ImageViewAbstract>,
    ) -> Arc<dyn DescriptorSet> {
        let mut desc_set_builder =
            PersistentDescriptorSet::start(pipeline.layout().descriptor_set_layouts()[0].clone());
        desc_set_builder.add_image(image_view).unwrap();
        Arc::new(desc_set_builder.build().unwrap())
    }
    #[inline(always)]
    fn get_descriptor_set(&self, texture_id: TextureId) -> Arc<dyn DescriptorSet> {
        let t_id = texture_id_as_option_u64(texture_id);
        self.egui_textures
            .get(&t_id)
            .expect("[BackEnd] invalid Texture ID given")
            .as_ref()
            .expect("[BackEnd] access to freed image")
            .0
            .clone()
    }
    #[inline(always)]
    fn set_texture(
        &mut self,
        id: TextureId,
        pixels: &[u8],
        size: (u32, u32),
        version: Option<u64>,
    ) {
        let format = match id {
            TextureId::Egui => {
                if self.egui_texture_version == version {
                    return;
                }
                self.egui_texture_version = version;
                vulkano::format::Format::R8_UNORM
            }
            TextureId::User(_) => vulkano::format::Format::R8G8B8A8_SRGB,
        };
        let staging_sub_buffer = self
            .image_staging_buffer
            .chunk(pixels.iter().copied())
            .unwrap();

        let (image, future) = vulkano::image::ImmutableImage::from_buffer(
            staging_sub_buffer,
            ImageDimensions::Dim2d {
                width: size.0,
                height: size.1,
                array_layers: 1,
            },
            MipmapsCount::One,
            format,
            self.queue.clone(),
        )
        .unwrap();
        let t_id = texture_id_as_option_u64(id);
        self.thread_pool.execute(move || future.flush().unwrap());
        let image_view = ImageView::new(image).unwrap() as Arc<dyn ImageViewAbstract>;
        let pipeline = self.pipeline.clone();
        let texture_desc = Self::create_descriptor_set_from_view(pipeline, image_view);
        self.egui_textures
            .insert(t_id, Some(TextureDescriptor(texture_desc)));
    }
    pub fn set_user_texture(
        &mut self,
        id: TextureId,
        size: (usize, usize),
        srgba_pixels: &[Color32],
    ) {
        assert!(
            (size.0 * size.1) >= srgba_pixels.len(),
            "Pixel count was invalid given {} x {} actual pixels {}",
            size.0,
            size.1,
            srgba_pixels.len()
        );
        let pixels = bytemuck::cast_slice(srgba_pixels);
        self.set_texture(id, pixels, (size.0 as u32, size.1 as u32), None);
    }
    /// register vulkano image view as egui texture
    ///
    /// Usable for render to image rectangle

    pub fn register_vulkano_image_view(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract>,
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
            vulkano::format::Format::R8G8B8A8_SRGB,
        )
    }
    pub fn init_vulkano_image_with_parameters(
        &mut self,
        dimensions: [u32; 2],
        usage: ImageUsage,
        format: vulkano::format::Format,
    ) -> Result<(TextureId, Arc<AttachmentImage>), InitRenderAreaError> {
        let pipeline = self.pipeline.clone();
        let image = AttachmentImage::with_usage(self.device.clone(), dimensions, format, usage)?;
        let texture_id = self.alloc_user_texture();
        let t_id = texture_id_as_option_u64(texture_id);
        self.egui_textures
            .get_mut(&t_id)
            .map(|slot| {
                let image_view = ImageView::new(image.clone())?;
                slot.replace(TextureDescriptor(Self::create_descriptor_set_from_view(
                    pipeline, image_view,
                )));
                Ok((texture_id, image))
            })
            .unwrap()
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
        let internal_id = texture_id_as_option_u64(texture_id);
        if let Some(slot) = self.egui_textures.get_mut(&internal_id) {
            //test we can retrieve format and usage from texture.

            let slot = if let Some(slot) = slot {
                slot
            } else {
                return Err(RecreateErrors::AlreadyFreed);
            };
            let image_access = slot.0.image(0).unwrap().0.image();
            let format = image_access.format();
            let usage = image_access.inner().image.usage();

            let image =
                AttachmentImage::with_usage(self.device.clone(), dimensions, format, usage)?;
            let image_view = ImageView::new(image.clone())?;

            let descriptor =
                Self::create_descriptor_set_from_view(self.pipeline.clone(), image_view);
            let texture_desc = TextureDescriptor(descriptor);
            // register new descriptor set
            *slot = texture_desc;
            Ok(image)
        } else {
            Err(RecreateErrors::TextureIdNotFound)
        }
    }
}
impl epi::NativeTexture for Painter {
    type Texture = Arc<dyn ImageViewAbstract>;

    fn register_native_texture(&mut self, native: Self::Texture) -> TextureId {
        self.register_vulkano_image_view(native)
    }

    fn replace_native_texture(&mut self, id: TextureId, replacing: Self::Texture) {
        if let egui::TextureId::User(x) = id {
            self.egui_textures
                .get_mut(&Some(x))
                .replace(&mut Some(TextureDescriptor(
                    Self::create_descriptor_set_from_view(self.pipeline.clone(), replacing),
                )));
        }
    }
}
impl epi::TextureAllocator for Painter {
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
used for 0 sized area erasing
*/
fn calc_scissor(
    clip_rect: egui::emath::Rect,
    screen_desc: ScreenDescriptor,
) -> Option<vulkano::pipeline::viewport::Scissor> {
    let pixels_per_point = screen_desc.scale_factor;
    let width_in_pixels = screen_desc.physical_width;
    let height_in_pixels = screen_desc.physical_height;
    /*
    shamelessly copy from egui_glium
    */
    // Transform clip rect to physical pixels:
    let clip_min_x = pixels_per_point * clip_rect.min.x;
    let clip_min_y = pixels_per_point * clip_rect.min.y;
    let clip_max_x = pixels_per_point * clip_rect.max.x;
    let clip_max_y = pixels_per_point * clip_rect.max.y;

    // Make sure clip rect can fit withing an `u32`:
    let clip_min_x = clip_min_x.clamp(0.0, width_in_pixels as f32);
    let clip_min_y = clip_min_y.clamp(0.0, height_in_pixels as f32);
    let clip_max_x = clip_max_x.clamp(clip_min_x, width_in_pixels as f32);
    let clip_max_y = clip_max_y.clamp(clip_min_y, height_in_pixels as f32);

    let clip_min_x = clip_min_x.round() as u32;
    let clip_min_y = clip_min_y.round() as u32;
    let clip_max_x = clip_max_x.round() as u32;
    let clip_max_y = clip_max_y.round() as u32;
    let width = clip_max_x - clip_min_x;
    let height = clip_max_y - clip_min_y;
    #[cfg(feature = "backend_debug")]
    println!(
        "[BackEnd] clip_rectangle min_x : {} min_y : {} width :{} height : {} ",
        clip_min_x, clip_min_y, width, height
    );
    if (width == 0) | (height == 0) {
        None
    } else {
        Some(Scissor {
            origin: [clip_min_x as u32, clip_min_y as u32],
            dimensions: [width as u32, height as u32],
        })
    }
}
pub(crate) struct TextureDescriptor(Arc<dyn DescriptorSet>);
pub enum GetDescriptorErrors {
    InvalidTextureId,
    FreedSlot,
}

#[derive(Debug, Clone)]
pub enum RecreateErrors {
    TextureIdIsEgui,
    TextureIdNotFound,
    AlreadyFreed,
    ImageViewCreationError(ImageViewCreationError),
    ImageCreationError(ImageCreationError),
}

impl From<ImageViewCreationError> for RecreateErrors {
    fn from(err: ImageViewCreationError) -> Self {
        Self::ImageViewCreationError(err)
    }
}
impl From<ImageCreationError> for RecreateErrors {
    fn from(err: ImageCreationError) -> Self {
        Self::ImageCreationError(err)
    }
}

#[derive(Debug, Clone)]
pub enum InitRenderAreaError {
    ImageViewCreationError(ImageViewCreationError),
    ImageCreationError(ImageCreationError),
}
impl From<ImageViewCreationError> for InitRenderAreaError {
    fn from(err: ImageViewCreationError) -> Self {
        Self::ImageViewCreationError(err)
    }
}
impl From<ImageCreationError> for InitRenderAreaError {
    fn from(err: ImageCreationError) -> Self {
        Self::ImageCreationError(err)
    }
}
