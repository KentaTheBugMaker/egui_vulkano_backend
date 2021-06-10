use crate::model::{Normal, Vertex};

use crate::model;
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use std::borrow::Cow;
use std::ffi::CString;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DynamicState, PrimaryAutoCommandBuffer,
    PrimaryCommandBuffer, SubpassContents,
};
use vulkano::descriptor::descriptor::{
    DescriptorBufferDesc, DescriptorDesc, DescriptorDescTy, ShaderStages,
};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::pipeline_layout::{PipelineLayoutDesc, PipelineLayoutDescPcRange};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageLayout, ImageUsage};
use vulkano::pipeline::shader::{
    GraphicsShaderType, ShaderInterfaceDef, ShaderInterfaceDefEntry, ShaderModule,
};
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::render_pass::{
    AttachmentDesc, Framebuffer, FramebufferAbstract, LoadOp, RenderPass, RenderPassDesc, StoreOp,
    Subpass, SubpassDesc,
};
use vulkano::sync::GpuFuture;

// shader interface definition
struct VSInput;
unsafe impl ShaderInterfaceDef for VSInput {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        /*
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
        */
        Box::new(
            [
                ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: Format::R32G32B32Sfloat,
                    name: Some(Cow::Borrowed("position")),
                },
                ShaderInterfaceDefEntry {
                    location: 1..2,
                    format: Format::R32G32B32Sfloat,
                    name: Some(Cow::Borrowed("normal")),
                },
            ]
            .iter()
            .cloned(),
        )
    }
}
//VS->FS interface
struct VSOutput;
unsafe impl ShaderInterfaceDef for VSOutput {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        /*layout(location = 0) out vec3 v_normal;*/
        Box::new(
            [ShaderInterfaceDefEntry {
                location: 0..1,
                format: Format::R32G32B32Sfloat,
                name: Some(Cow::Borrowed("v_normal")),
            }]
            .iter()
            .cloned(),
        )
    }
}

struct FSOutput;
unsafe impl ShaderInterfaceDef for FSOutput {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        /*layout(location = 0) out vec4 f_color;*/
        Box::new(
            [ShaderInterfaceDefEntry {
                location: 0..1,
                format: Format::R32G32B32A32Sfloat,
                name: Some(Cow::Borrowed("f_color")),
            }]
            .iter()
            .cloned(),
        )
    }
}
//render pass
fn create_renderpass() -> RenderPassDesc {
    let color = AttachmentDesc {
        format: Format::R8G8B8A8Srgb,
        samples: 1,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        stencil_load: LoadOp::Load,
        stencil_store: StoreOp::Store,
        initial_layout: ImageLayout::ColorAttachmentOptimal,
        final_layout: ImageLayout::ColorAttachmentOptimal,
    };
    let depth = AttachmentDesc {
        format: Format::D32Sfloat,
        samples: 1,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        stencil_load: LoadOp::Load,
        stencil_store: StoreOp::Store,
        initial_layout: ImageLayout::DepthStencilAttachmentOptimal,
        final_layout: ImageLayout::DepthStencilAttachmentOptimal,
    };
    let sub_pass_desc = SubpassDesc {
        color_attachments: vec![(0, ImageLayout::ColorAttachmentOptimal)],
        depth_stencil: Some((1, ImageLayout::DepthStencilAttachmentOptimal)),
        input_attachments: vec![],
        resolve_attachments: vec![],
        preserve_attachments: vec![],
    };
    RenderPassDesc::new(vec![color, depth], vec![sub_pass_desc], vec![])
}
#[derive(Default, Debug, Copy, Clone)]
struct PipelineLayout;
unsafe impl PipelineLayoutDesc for PipelineLayout {
    fn num_sets(&self) -> usize {
        1
    }

    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        if set == 0 {
            Some(1)
        } else {
            None
        }
    }

    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        if set == 0 && binding == 0 {
            Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: None,
                    storage: false,
                }),
                array_count: 1,
                stages: ShaderStages {
                    vertex: true,
                    tessellation_control: false,
                    tessellation_evaluation: false,
                    geometry: false,
                    fragment: false,
                    compute: false,
                },
                readonly: true,
            })
        } else {
            None
        }
    }

    fn num_push_constants_ranges(&self) -> usize {
        0
    }

    fn push_constants_range(&self, _num: usize) -> Option<PipelineLayoutDescPcRange> {
        None
    }
}
fn create_pipeline(
    device: Arc<Device>,
) -> Arc<
    GraphicsPipeline<
        TwoBuffersDefinition<Vertex, Normal>,
        Box<dyn PipelineLayoutAbstract + Send + Sync>,
    >,
> {
    let vs = unsafe { ShaderModule::new(device.clone(), include_bytes!("vert.spv")) }.unwrap();
    let fs = unsafe { ShaderModule::new(device.clone(), include_bytes!("frag.spv")) }.unwrap();
    let cstring = CString::new("main").unwrap();
    let vs_entry = unsafe {
        vs.graphics_entry_point(
            &cstring,
            VSInput,
            VSOutput,
            PipelineLayout,
            GraphicsShaderType::Vertex,
        )
    };
    let fs_entry = unsafe {
        fs.graphics_entry_point(
            &cstring,
            VSOutput,
            FSOutput,
            PipelineLayout,
            GraphicsShaderType::Fragment,
        )
    };
    let render_pass = Arc::new(RenderPass::new(device.clone(), create_renderpass()).unwrap());
    Arc::new(
        GraphicsPipeline::start()
            .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
            .vertex_shader(vs_entry, ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs_entry, ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .build(device)
            .unwrap(),
    )
}
//Renderer
pub struct TeapotRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    normal_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    render_pass: Arc<RenderPass>,
    frame_buffer: Option<Arc<dyn FramebufferAbstract + Send + Sync>>,
    uniform_uploading_buffer_pool: CpuBufferPool<Uniforms>,
    rotate: f32,
    aspect_ratio: f32,
}
#[repr(C)]
pub struct Uniforms {
    pub world: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}
impl TeapotRenderer {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> TeapotRenderer {
        let pipeline = create_pipeline(device.clone());
        let render_pass = pipeline.render_pass().clone();
        let uniforms_uploading_buffer_pool =
            CpuBufferPool::new(device.clone(), BufferUsage::uniform_buffer());
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            model::VERTICES.iter().copied(),
        )
        .unwrap();
        let normal_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            model::NORMALS.iter().copied(),
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::index_buffer(),
            false,
            model::INDICES.iter().copied(),
        )
        .unwrap();
        Self {
            device,
            queue,
            vertex_buffer,
            index_buffer,
            normal_buffer,
            pipeline: pipeline as Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
            render_pass,
            frame_buffer: None,
            uniform_uploading_buffer_pool: uniforms_uploading_buffer_pool,
            rotate: 0.0,
            aspect_ratio: 0.0,
        }
    }
    pub fn set_render_target(&mut self, rt: Arc<AttachmentImage>) {
        //create depth
        let depth = AttachmentImage::with_usage(
            self.device.clone(),
            rt.dimensions(),
            Format::D32Sfloat,
            ImageUsage::depth_stencil_attachment(),
        )
        .unwrap();
        //create framebuffer
        let frame_buffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(ImageView::new(rt.clone()).unwrap())
                .unwrap()
                .add(ImageView::new(depth).unwrap())
                .unwrap()
                .build()
                .unwrap(),
        ) as Arc<dyn FramebufferAbstract + Send + Sync>;
        self.frame_buffer.replace(frame_buffer);
        self.aspect_ratio = rt.dimensions()[0] as f32 / rt.dimensions()[1] as f32;
    }
    pub fn draw(&mut self) {
        if let Some(frame_buffer) = self.frame_buffer.as_ref() {
            let uniform_buffer_sub_buffer = {
                let proj = cgmath::perspective(
                    Rad(std::f32::consts::FRAC_PI_2),
                    self.aspect_ratio,
                    0.01,
                    100.0,
                );
                let rotation = Matrix3::from_angle_y(Rad(self.rotate));
                let view = Matrix4::look_at_rh(
                    Point3::new(0.3, 0.3, 1.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, -1.0, 0.0),
                );
                let scale = Matrix4::from_scale(0.01);
                let data = Uniforms {
                    world: Matrix4::from(rotation).into(),
                    view: (view * scale).into(),
                    proj: proj.into(),
                };
                self.uniform_uploading_buffer_pool.next(data).unwrap()
            };
            let layout = self.pipeline.descriptor_set_layout(0).unwrap().clone();
            let set = Arc::new(
                PersistentDescriptorSet::start(layout)
                    .add_buffer(uniform_buffer_sub_buffer)
                    .unwrap()
                    .build()
                    .unwrap(),
            );
            let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            let dynamic_state = DynamicState {
                line_width: None,
                viewports: Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [frame_buffer.width() as f32, frame_buffer.height() as f32],
                    depth_range: 0.0..1.0,
                }]),
                scissors: None,
                compare_mask: None,
                write_mask: None,
                reference: None,
            };
            command_buffer_builder
                .begin_render_pass(
                    frame_buffer.clone(),
                    SubpassContents::Inline,
                    vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                )
                .unwrap()
                .draw_indexed(
                    self.pipeline.clone(),
                    &dynamic_state,
                    vec![self.vertex_buffer.clone(), self.normal_buffer.clone()],
                    self.index_buffer.clone(),
                    set,
                    (),
                    vec![],
                )
                .unwrap()
                .end_render_pass()
                .unwrap();
            let command_buffer: PrimaryAutoCommandBuffer = command_buffer_builder.build().unwrap();
            command_buffer
                .execute(self.queue.clone())
                .unwrap()
                .cleanup_finished();
        } else {
            panic!("Frame buffer is blank");
        }
    }
    pub fn set_rotate(&mut self, rot: f32) {
        self.rotate = rot;
    }
}
