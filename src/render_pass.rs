use vulkano::image::ImageLayout;
use vulkano::render_pass::{AttachmentDesc, LoadOp, RenderPassDesc, StoreOp, SubpassDesc};

use vulkano::format::Format;

#[derive(Copy, Clone)]
pub struct EguiRenderPassDesc {
    pub color: (Format, u32),
}
pub(crate) fn render_pass_desc_from_format(format: Format) -> RenderPassDesc {
    let attachment_desc = AttachmentDesc {
        format,
        samples: 1,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        stencil_load: LoadOp::Clear,
        stencil_store: StoreOp::Store,
        initial_layout: ImageLayout::ColorAttachmentOptimal,
        final_layout: ImageLayout::ColorAttachmentOptimal,
    };
    let sub_pass_desc = SubpassDesc {
        color_attachments: vec![(0, ImageLayout::ColorAttachmentOptimal)],
        depth_stencil: None,
        input_attachments: vec![],
        resolve_attachments: vec![],
        preserve_attachments: vec![],
    };
    RenderPassDesc::new(vec![attachment_desc], vec![sub_pass_desc], vec![])
}
