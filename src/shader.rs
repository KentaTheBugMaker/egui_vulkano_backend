use std::ffi::CStr;
use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor, BlendOp};
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::layout::PipelineLayoutPcRange;
use vulkano::pipeline::shader::{
    GraphicsShaderType, ShaderInterface, ShaderInterfaceEntry, ShaderModule, ShaderStages,
    SpecializationConstants,
};
use vulkano::render_pass::{RenderPass, Subpass};

use crate::painter::{EguiVulkanoVertex, Pipeline, PushConstants};
use crate::render_pass::render_pass_desc_from_format;
use log::debug;
use vulkano::descriptor_set::layout::{
    DescriptorDesc, DescriptorDescTy, DescriptorImageDesc, DescriptorImageDescArray,
    DescriptorImageDescDimensions, DescriptorSetDesc,
};

pub(crate) fn create_pipeline(device: Arc<Device>, render_target_format: Format) -> Arc<Pipeline> {
    //this is safe because we use offline compiled shader binary and shipped with this backend
    debug!("start create shader modules");
    let vs_module =
        unsafe { ShaderModule::new(device.clone(), include_bytes!("shaders/vert.spv")) }.unwrap();
    debug!("vs_module created");
    let fs_module =
        unsafe { ShaderModule::new(device.clone(), include_bytes!("shaders/frag.spv")) }.unwrap();
    debug!("fs_module created");
    /*
    layout(set = 1, binding = 0) uniform texture2D t_texture;
    layout(set = 0, binding = 0) uniform sampler s_texture;
    layout(push_constant) uniform UniformBuffer {
        vec2 u_screen_size;
        bool is_egui_system_texture;
    };
    */
    let pc_range = Some(PipelineLayoutPcRange {
        offset: 0,
        size: std::mem::size_of::<PushConstants>(),
        stages: ShaderStages {
            vertex: true,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: true,
            compute: false,
        },
    });
    debug!("pc_range created");
    let descriptor_set_desc_0 = DescriptorSetDesc::new(vec![Some(DescriptorDesc {
        ty: DescriptorDescTy::Sampler,
        array_count: 1,
        stages: ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: true,
            compute: false,
        },
        readonly: true,
    })]);
    debug!("descriptor set desc 0 created");
    let descriptor_set_desc_1 = DescriptorSetDesc::new(vec![Some(DescriptorDesc {
        ty: DescriptorDescTy::Image(DescriptorImageDesc {
            sampled: true,
            dimensions: DescriptorImageDescDimensions::TwoDimensional,
            format: None,
            multisampled: false,
            array_layers: DescriptorImageDescArray::NonArrayed,
        }),
        array_count: 1,
        stages: ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: true,
            compute: false,
        },
        readonly: true,
    })]);
    debug!("descriptor set desc 1 created");
    let descriptor_sets = vec![descriptor_set_desc_0, descriptor_set_desc_1];
    debug!("descriptor  set created");
    /*
        # SAFETY
        [x] one entry per location
        [x] each format must not be larger than 128bit
    */
    let vs_in = unsafe {
        ShaderInterface::new_unchecked(vec![
            ShaderInterfaceEntry {
                location: 0..1,
                format: Format::R32G32Sfloat,
                name: Some(std::borrow::Cow::Borrowed("a_pos")),
            },
            ShaderInterfaceEntry {
                location: 1..2,
                format: Format::R32G32Sfloat,
                name: Some(std::borrow::Cow::Borrowed("a_tex_coord")),
            },
            ShaderInterfaceEntry {
                location: 2..3,
                format: Format::R32Uint,
                name: Some(std::borrow::Cow::Borrowed("a_color")),
            },
        ])
    };
    debug!("vs input shader interface created");
    let vs_out = unsafe {
        ShaderInterface::new_unchecked(vec![
            ShaderInterfaceEntry {
                location: 0..1,
                format: Format::R32G32Sfloat,
                name: Some("v_tex_coord".into()),
            },
            ShaderInterfaceEntry {
                location: 1..2,
                format: Format::R32G32B32A32Sfloat,
                name: Some("v_color".into()),
            },
        ])
    };
    debug!("vs output fs input shader interface created");
    let fs_out = unsafe {
        ShaderInterface::new_unchecked(vec![ShaderInterfaceEntry {
            location: 0..1,
            format: Format::R32G32B32A32Sfloat,
            name: Some(std::borrow::Cow::Borrowed("f_color")),
        }])
    };
    debug!("fs output shader interface created");
    let vs_entry = unsafe {
        vs_module.graphics_entry_point(
            CStr::from_bytes_with_nul_unchecked(b"main\0"),
            descriptor_sets.clone(),
            pc_range,
            <()>::descriptors(),
            vs_in,
            vs_out.clone(),
            GraphicsShaderType::Vertex,
        )
    };
    debug!("vs entry created");
    let fs_entry = unsafe {
        fs_module.graphics_entry_point(
            CStr::from_bytes_with_nul_unchecked(b"main\0"),
            descriptor_sets,
            pc_range,
            <()>::descriptors(),
            vs_out,
            fs_out,
            GraphicsShaderType::Fragment,
        )
    };
    debug!("fs entry created");
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
            .vertex_input_single_buffer::<EguiVulkanoVertex>()
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
