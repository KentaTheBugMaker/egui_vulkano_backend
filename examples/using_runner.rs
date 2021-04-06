/// from epi limitation we can not use offscreen rendering
///
fn main() {
    let app = Box::new(egui_demo_lib::WrapApp::default());
    egui_vulkano_backend::runner::run(app);
}
