use std::borrow::Cow;
use std::ffi::CString;
use std::sync::Arc;

use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, PrimaryCommandBuffer,
    SubpassContents,
};

use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageLayout, ImageUsage, SampleCount};

use vulkano::pipeline::shader::{
    GraphicsShaderType, ShaderInterface, ShaderInterfaceEntry, ShaderModule, ShaderStages,
};
use vulkano::pipeline::vertex::BuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{
    AttachmentDesc, Framebuffer, FramebufferAbstract, LoadOp, RenderPass, RenderPassDesc, StoreOp,
    Subpass, SubpassDesc,
};
use vulkano::sync::GpuFuture;

use crate::model;
use crate::model::{Normal, Vertex};
use vulkano::descriptor_set::layout::{DescriptorDesc, DescriptorDescTy, DescriptorSetDesc};
use vulkano::descriptor_set::PersistentDescriptorSet;
// shader interface definition

//pipeline_layout_desc
fn create_descriptor_set_desc() -> DescriptorSetDesc {
    DescriptorSetDesc::new(vec![Some(DescriptorDesc {
        ty: DescriptorDescTy::UniformBuffer,
        descriptor_count: 1,
        stages: ShaderStages {
            vertex: true,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: false,
            compute: false,
        },
        variable_count: false,
        mutable: false,
    })])
}

//render pass
fn create_renderpass() -> RenderPassDesc {
    let color = AttachmentDesc {
        format: Format::R8G8B8A8_SRGB,
        samples: SampleCount::Sample1,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        stencil_load: LoadOp::Load,
        stencil_store: StoreOp::Store,
        initial_layout: ImageLayout::ColorAttachmentOptimal,
        final_layout: ImageLayout::ColorAttachmentOptimal,
    };
    let depth = AttachmentDesc {
        format: Format::D32_SFLOAT,
        samples: SampleCount::Sample1,
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

fn create_pipeline(device: Arc<Device>) -> Arc<GraphicsPipeline> {
    let vs = unsafe { ShaderModule::new(device.clone(), include_bytes!("vert.spv")) }.unwrap();
    let fs = unsafe { ShaderModule::new(device.clone(), include_bytes!("frag.spv")) }.unwrap();
    let cstring = CString::new("main").unwrap();
    let empty_spec_constant = &[];
    let vs_in = unsafe {
        ShaderInterface::new_unchecked(vec![
            ShaderInterfaceEntry {
                location: 0..1,
                format: Format::R32G32B32_SFLOAT,
                name: Some(Cow::Borrowed("position")),
            },
            ShaderInterfaceEntry {
                location: 1..2,
                format: Format::R32G32B32_SFLOAT,
                name: Some(Cow::Borrowed("normal")),
            },
        ])
    };
    let vs_out = unsafe {
        ShaderInterface::new_unchecked(vec![ShaderInterfaceEntry {
            location: 0..1,
            format: Format::R32G32B32_SFLOAT,
            name: Some(Cow::Borrowed("v_normal")),
        }])
    };
    let fs_out = unsafe {
        ShaderInterface::new_unchecked(vec![ShaderInterfaceEntry {
            location: 0..1,
            format: Format::R32G32B32A32_SFLOAT,
            name: Some(Cow::Borrowed("f_color")),
        }])
    };
    let descriptor_set_descs = vec![create_descriptor_set_desc()];
    let vs_entry = unsafe {
        vs.graphics_entry_point(
            &cstring,
            descriptor_set_descs.clone(),
            None,
            empty_spec_constant,
            vs_in,
            vs_out.clone(),
            GraphicsShaderType::Vertex,
        )
    };
    let fs_entry = unsafe {
        fs.graphics_entry_point(
            &cstring,
            descriptor_set_descs,
            None,
            empty_spec_constant,
            vs_out,
            fs_out,
            GraphicsShaderType::Fragment,
        )
    };
    let render_pass = Arc::new(RenderPass::new(device.clone(), create_renderpass()).unwrap());
    Arc::new(
        GraphicsPipeline::start()
            .vertex_input(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .vertex::<Normal>(),
            )
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
    pipeline: Arc<GraphicsPipeline>,
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
        let render_pass = pipeline.subpass().render_pass().clone();
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
            pipeline,
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
            [rt.dimensions()[0], rt.dimensions()[1]],
            Format::D32_SFLOAT,
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
            let layout = self.pipeline.layout().descriptor_set_layouts()[0].clone();
            let mut desc_set_builder = PersistentDescriptorSet::start(layout);
            desc_set_builder
                .add_buffer(Arc::new(uniform_buffer_sub_buffer))
                .unwrap();

            let set = Arc::new(desc_set_builder.build().unwrap());
            let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            let view_ports = vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [frame_buffer.width() as f32, frame_buffer.height() as f32],
                depth_range: 0.0..1.0,
            }];

            command_buffer_builder
                .begin_render_pass(
                    frame_buffer.clone(),
                    SubpassContents::Inline,
                    vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                )
                .unwrap()
                .bind_pipeline_graphics(self.pipeline.clone())
                .set_viewport(0, view_ports)
                .bind_vertex_buffers(0, (self.vertex_buffer.clone(), self.normal_buffer.clone()))
                .bind_index_buffer(self.index_buffer.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
                .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
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
