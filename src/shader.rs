use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor, BlendOp};
use vulkano::pipeline::input_assembly::PrimitiveTopology;

use vulkano::render_pass::{RenderPass, Subpass};

use crate::painter::{PushConstants, WrappedEguiVertex};
use crate::render_pass::render_pass_desc_from_format;
use log::debug;
use std::ffi::CString;
use vulkano::descriptor_set::layout::{
    DescriptorDesc, DescriptorDescImage, DescriptorDescTy, DescriptorSetDesc,
};
use vulkano::image::view::ImageViewType;
use vulkano::pipeline::layout::PipelineLayoutPcRange;
use vulkano::pipeline::shader::{
    GraphicsShaderType, ShaderInterface, ShaderInterfaceEntry, ShaderModule, ShaderStages,
};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};

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
    let vs_module =
        unsafe { ShaderModule::new(device.clone(), include_bytes!("shaders/vert.spv")) }.unwrap();
    let fs_module =
        unsafe { ShaderModule::new(device.clone(), include_bytes!("shaders/frag.spv")) }.unwrap();
    let main = CString::new("main").unwrap();
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
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: true,
            compute: false,
        },
    });

    let descriptor_set_desc_0 = DescriptorSetDesc::new(vec![Some(DescriptorDesc {
        ty: DescriptorDescTy::CombinedImageSampler {
            image_desc: DescriptorDescImage {
                format: None,
                multisampled: false,
                view_type: ImageViewType::Dim2d,
            },
            immutable_samplers: vec![sampler],
        },
        descriptor_count: 1,
        stages: ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: true,
            compute: false,
        },
        variable_count: false,
        mutable: false,
    })]);
    let descriptor_sets = vec![descriptor_set_desc_0];
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
    let spec = &[];
    let vs_entry = unsafe {
        vs_module.graphics_entry_point(
            &main,
            descriptor_sets.clone(),
            pc_range,
            spec,
            vs_in,
            vs_out.clone(),
            GraphicsShaderType::Vertex,
        )
    };

    let fs_entry = unsafe {
        fs_module.graphics_entry_point(
            &main,
            descriptor_sets,
            pc_range,
            spec,
            vs_out,
            fs_out,
            GraphicsShaderType::Fragment,
        )
    };
    let render_pass = Arc::new(
        RenderPass::new(
            device.clone(),
            render_pass_desc_from_format(render_target_format),
        )
        .unwrap(),
    );

    debug!("renderpass created");
    let pipeline = Arc::new({
        vulkano::pipeline::GraphicsPipeline::start()
            .viewports_scissors_dynamic(1)
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .fragment_shader(fs_entry, ())
            .vertex_input_single_buffer::<WrappedEguiVertex>()
            .vertex_shader(vs_entry, ())
            .primitive_topology(PrimitiveTopology::TriangleList)
            .front_face_clockwise()
            .polygon_mode_fill()
            .depth_stencil_disabled()
            .blend_collective(AttachmentBlend {
                enabled: true,
                color_op: BlendOp::Add,
                color_source: BlendFactor::One,
                color_destination: BlendFactor::OneMinusSrcAlpha,
                alpha_op: BlendOp::Add,
                alpha_source: BlendFactor::OneMinusDstAlpha,
                alpha_destination: BlendFactor::One,
                mask_red: true,
                mask_green: true,
                mask_blue: true,
                mask_alpha: true,
            })
            .build(device)
            .unwrap()
    });
    debug!("pipeline created");
    pipeline
}
