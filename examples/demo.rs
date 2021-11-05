use epi::NativeOptions;

fn main() {
    let app = egui_demo_lib::WrapApp::default();
    egui_vulkano_backend::eframe::run(
        Box::new(app),
        &NativeOptions {
            drag_and_drop_support: true,
            resizable: true,
            transparent: true,
            ..Default::default()
        },
    );
}
