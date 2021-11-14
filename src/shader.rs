use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;

use vulkano::pipeline::input_assembly::{InputAssemblyState, PrimitiveTopology};

use vulkano::render_pass::{RenderPass, Subpass};

use crate::painter::{PushConstants, WrappedEguiVertex};
use crate::render_pass::render_pass_desc_from_format;
use log::debug;

use vulkano::pipeline::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
    ColorComponents,
};
use vulkano::pipeline::depth_stencil::DepthStencilState;

use vulkano::pipeline::rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState};

use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::image::view::ImageViewType;
use vulkano::pipeline::layout::PipelineLayoutPcRange;
use vulkano::pipeline::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, StateMode};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::shader::{
    DescriptorRequirements, EntryPointInfo, ShaderExecution, ShaderInterface, ShaderInterfaceEntry,
    ShaderModule, ShaderStages,
};
use vulkano::Version;

use fnv::FnvHashMap;

pub(crate) fn create_pipeline(
    device: Arc<Device>,
    render_target_format: Format,
) -> Arc<GraphicsPipeline> {
    //this is safe because we use offline compiled shader binary and shipped with this backend
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
    .expect("failed to create sampler");
    /*
    layout(set = 0, binding = 0) uniform sampler2D t_texture;
    layout(push_constant) uniform UniformBuffer {
        vec2 u_screen_size;
        bool is_egui_system_texture;
    };
    */
    let pc_range = Some(PipelineLayoutPcRange {
        offset: 0,
        size: std::mem::size_of::<PushConstants>() as u32,
        stages: ShaderStages {
            vertex: true,
            fragment: true,
            ..ShaderStages::none()
        },
    });
    /*
        # SAFETY
        [x] one entry per location
        [x] each format must not be larger than 128bit
    */
    let vs_in = unsafe {
        ShaderInterface::new_unchecked(vec![
            ShaderInterfaceEntry {
                location: 0..1,
                format: Format::R32G32_SFLOAT,
                name: Some(std::borrow::Cow::Borrowed("a_pos")),
            },
            ShaderInterfaceEntry {
                location: 1..2,
                format: Format::R32G32_SFLOAT,
                name: Some(std::borrow::Cow::Borrowed("a_tex_coord")),
            },
            ShaderInterfaceEntry {
                location: 2..3,
                format: Format::R32_UINT,
                name: Some(std::borrow::Cow::Borrowed("a_color")),
            },
        ])
    };
    let vs_out = unsafe {
        ShaderInterface::new_unchecked(vec![
            ShaderInterfaceEntry {
                location: 0..1,
                format: Format::R32G32_SFLOAT,
                name: Some("v_tex_coord".into()),
            },
            ShaderInterfaceEntry {
                location: 1..2,
                format: Format::R32G32B32A32_SFLOAT,
                name: Some("v_color".into()),
            },
        ])
    };
    let fs_out = unsafe {
        ShaderInterface::new_unchecked(vec![ShaderInterfaceEntry {
            location: 0..1,
            format: Format::R32G32B32A32_SFLOAT,
            name: Some(std::borrow::Cow::Borrowed("f_color")),
        }])
    };
    let descriptor = DescriptorRequirements {
        descriptor_types: vec![DescriptorType::CombinedImageSampler],
        descriptor_count: 1,
        format: None,
        image_view_type: Some(ImageViewType::Dim2d),
        multisampled: false,
        mutable: false,
        stages: ShaderStages {
            fragment: true,
            ..Default::default()
        },
    };
    let vs_module = unsafe {
        ShaderModule::from_bytes_with_data(
            device.clone(),
            include_bytes!("shaders/vert.spv"),
            Version::V1_0,
            [],
            [],
            [(
                "main".to_owned(),
                EntryPointInfo {
                    execution: ShaderExecution::Vertex,
                    descriptor_requirements: Default::default(),
                    push_constant_requirements: pc_range,
                    specialization_constant_requirements: Default::default(),
                    input_interface: vs_in,
                    output_interface: vs_out.clone(),
                },
            )],
        )
    }
    .unwrap();
    let mut descriptor_requirements: FnvHashMap<(u32, u32), DescriptorRequirements> =
        Default::default();
    descriptor_requirements.insert((0, 0), descriptor);
    let fs_module = unsafe {
        ShaderModule::from_bytes_with_data(
            device.clone(),
            include_bytes!("shaders/frag.spv"),
            Version::V1_0,
            [],
            [],
            [(
                "main".to_owned(),
                EntryPointInfo {
                    execution: ShaderExecution::Fragment,
                    descriptor_requirements,
                    push_constant_requirements: pc_range,
                    specialization_constant_requirements: Default::default(),
                    input_interface: vs_out,
                    output_interface: fs_out,
                },
            )],
        )
    }
    .unwrap();

    let render_pass = RenderPass::new(
        device.clone(),
        render_pass_desc_from_format(render_target_format),
    )
    .unwrap();
    debug!("renderpass created");
    let pipeline = {
        vulkano::pipeline::GraphicsPipeline::start()
            .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .fragment_shader(fs_module.entry_point("main").unwrap(), ())
            .vertex_input_single_buffer::<WrappedEguiVertex>()
            .vertex_shader(vs_module.entry_point("main").unwrap(), ())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
            )
            .rasterization_state(
                RasterizationState::new()
                    .cull_mode(CullMode::None)
                    .front_face(FrontFace::CounterClockwise)
                    .polygon_mode(PolygonMode::Fill),
            )
            .depth_stencil_state(DepthStencilState::disabled())
            .color_blend_state(ColorBlendState {
                logic_op: None,
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend {
                        color_op: BlendOp::Add,
                        color_source: BlendFactor::One,
                        color_destination: BlendFactor::OneMinusSrcAlpha,
                        alpha_op: BlendOp::Add,
                        alpha_source: BlendFactor::OneMinusDstAlpha,
                        alpha_destination: BlendFactor::One,
                    }),
                    color_write_mask: ColorComponents {
                        r: true,
                        g: true,
                        b: true,
                        a: true,
                    },
                    color_write_enable: StateMode::Fixed(true),
                }],
                blend_constants: StateMode::Fixed([1.0, 1.0, 1.0, 1.0]),
            })
            .with_auto_layout(device, |x| {
                x[0].set_immutable_samplers(0, [sampler]);
            })
            .unwrap()
    };
    debug!("pipeline created");
    pipeline
}
