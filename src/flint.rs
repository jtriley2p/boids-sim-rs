//! Flint design system for egui: tokens, fonts and widget style.

use egui::{
    Color32, CornerRadius, FontData, FontFamily, FontId, RichText, Stroke, TextStyle,
    epaint::{Shadow, text::{FontTweak, VariationCoords}},
    style::{HandleShape, Selection, WidgetVisuals},
};

const fn rgb(c: u32) -> Color32 {
    Color32::from_rgb((c >> 16) as u8, (c >> 8) as u8, c as u8)
}

/// `ink` at a fixed strength, for strokes and tints.
const fn ink(alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied_const(0xec, 0xe9, 0xe4, alpha)
}

pub const SURFACE_SUNKEN: Color32 = rgb(0x141110);
pub const SURFACE: Color32 = rgb(0x1c1917);
pub const SURFACE_RAISED: Color32 = rgb(0x221f1d);
pub const SURFACE_OVERLAY: Color32 = rgb(0x2a2624);
pub const TINT_HOVER: Color32 = ink(0x0d);
pub const TINT_ACTIVE: Color32 = ink(0x17);
pub const STROKE_SUBTLE: Color32 = ink(0x1a);
pub const STROKE: Color32 = ink(0x47);
pub const STROKE_STRONG: Color32 = ink(0x70);
pub const INK: Color32 = rgb(0xece9e4);
pub const INK_MUTED: Color32 = rgb(0xb6b2ad);
pub const INK_SUBTLE: Color32 = rgb(0x9c9793);
pub const DANGER: Color32 = rgb(0xf66d64);
pub const WARNING: Color32 = rgb(0xeec060);

pub const SPACE_1: f32 = 4.0;
pub const SPACE_2: f32 = 8.0;
pub const SPACE_3: f32 = 12.0;
pub const SPACE_4: f32 = 16.0;
pub const BORDER_1: f32 = 2.0;
pub const BORDER_2: f32 = 4.0;
/// Height of a small control.
pub const CONTROL_SM: f32 = 28.0;

fn family(name: &str) -> FontFamily {
    FontFamily::Name(name.into())
}

/// Flint `label` text: mono caps, tracked 0.04em, `ink-muted`.
pub fn label(text: &str) -> RichText {
    RichText::new(text.to_uppercase())
        .text_style(TextStyle::Name("label".into()))
        .extra_letter_spacing(0.04 * 11.0)
        .color(INK_MUTED)
}

/// Install Flint's fonts and style. Flint is dark only.
pub fn apply(ctx: &egui::Context) {
    let mono: &[u8] = include_bytes!("../assets/fonts/CascadiaMono-Variable.ttf");
    let medium = FontTweak { coords: VariationCoords::new([(b"wght", 500.0)]), ..Default::default() };
    let faces = [
        ("chakra", FontFamily::Proportional, FontData::from_static(include_bytes!("../assets/fonts/ChakraPetch-Regular-latin.ttf"))),
        ("chakra-medium", family("medium"), FontData::from_static(include_bytes!("../assets/fonts/ChakraPetch-Medium-latin.ttf"))),
        ("chakra-semibold", family("semibold"), FontData::from_static(include_bytes!("../assets/fonts/ChakraPetch-SemiBold-latin.ttf"))),
        ("cascadia", FontFamily::Monospace, FontData::from_static(mono)),
        ("cascadia-medium", family("label"), FontData::from_static(mono).tweak(medium)),
    ];
    // egui's bundled fonts stay behind ours as glyph fallbacks
    let mut fonts = egui::FontDefinitions::default();
    for (name, fam, data) in faces {
        fonts.font_data.insert(name.into(), data.into());
        fonts.families.entry(fam).or_default().insert(0, name.into());
    }
    ctx.set_fonts(fonts);

    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |s| {
        s.text_styles = [
            (TextStyle::Heading, FontId::new(17.0, family("semibold"))),
            (TextStyle::Body, FontId::new(15.0, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(14.0, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(14.0, family("medium"))),
            (TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace)),
            (TextStyle::Name("label".into()), FontId::new(11.0, family("label"))),
        ]
        .into();
        s.drag_value_text_style = TextStyle::Monospace;
        s.animation_time = 0.0; // overlays appear instantly, nothing moves
        s.interaction.selectable_labels = false;

        s.spacing.item_spacing = egui::vec2(SPACE_2, SPACE_1);
        s.spacing.button_padding = egui::vec2(SPACE_2, 0.0);
        s.spacing.interact_size = egui::vec2(56.0, CONTROL_SM);
        s.spacing.slider_rail_height = BORDER_1;

        let v = &mut s.visuals;
        // egui's windows, menus and tooltips are Flint overlays
        v.window_fill = SURFACE_OVERLAY;
        v.window_stroke = Stroke::new(BORDER_1, STROKE_STRONG);
        v.window_corner_radius = CornerRadius::ZERO;
        v.menu_corner_radius = CornerRadius::ZERO;
        v.window_shadow = Shadow::NONE;
        v.popup_shadow = Shadow::NONE;
        v.panel_fill = SURFACE;
        v.extreme_bg_color = SURFACE_SUNKEN; // input wells
        v.code_bg_color = SURFACE_SUNKEN;
        v.selection = Selection { bg_fill: STROKE, stroke: Stroke::new(BORDER_2, INK) }; // focused input: doubled edge
        v.text_cursor.stroke = Stroke::new(BORDER_1, INK);
        v.hyperlink_color = INK;
        v.warn_fg_color = WARNING;
        v.error_fg_color = DANGER;
        v.handle_shape = HandleShape::Rect { aspect_ratio: 0.4 };
        v.slider_trailing_fill = false;
        v.interact_cursor = Some(egui::CursorIcon::PointingHand);

        // secondary button at rest; `bg_fill` is the slider rail, opaque so the handle overlaps it cleanly
        let rest = WidgetVisuals {
            bg_fill: SURFACE_RAISED.blend(STROKE_STRONG),
            weak_bg_fill: Color32::TRANSPARENT,
            bg_stroke: Stroke::new(BORDER_1, STROKE_STRONG),
            corner_radius: CornerRadius::ZERO,
            fg_stroke: Stroke::new(BORDER_1, INK),
            expansion: 0.0,
        };
        let w = &mut v.widgets;
        w.noninteractive = WidgetVisuals {
            bg_stroke: Stroke::new(BORDER_1, STROKE_SUBTLE), // rules inside a surface
            fg_stroke: Stroke::new(BORDER_1, INK_MUTED),     // body copy
            ..rest
        };
        w.inactive = rest;
        w.hovered = WidgetVisuals { bg_fill: INK, weak_bg_fill: TINT_HOVER, bg_stroke: Stroke::new(BORDER_1, INK), ..rest };
        // egui shares one state for pressed and keyboard focus, so it carries the focus edge
        w.active = WidgetVisuals { bg_fill: INK, weak_bg_fill: TINT_ACTIVE, bg_stroke: Stroke::new(BORDER_2, INK), ..rest };
        w.open = w.active;
    });
}
