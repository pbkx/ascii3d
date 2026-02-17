use crate::{targets::buffer::BufferTarget, types::Rgb8};
use crossterm::{
    cursor,
    style::{self, Color},
    terminal,
};
use std::io::{self, Write};

pub struct TerminalGuard {
    active: bool,
}

impl TerminalGuard {
    pub fn new() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        let mut out = io::stdout();
        if let Err(err) = crossterm::execute!(
            out,
            terminal::EnterAlternateScreen,
            cursor::Hide,
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        ) {
            let _ = crossterm::execute!(out, cursor::Show, terminal::LeaveAlternateScreen);
            let _ = terminal::disable_raw_mode();
            return Err(err);
        }
        Ok(Self { active: true })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let _ = crossterm::execute!(io::stdout(), cursor::Show, terminal::LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
        self.active = false;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Auto,
    Truecolor,
    Ansi256,
    Mono,
}

impl Default for ColorMode {
    fn default() -> Self {
        ColorMode::Auto
    }
}

impl std::fmt::Display for ColorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ColorMode::Auto => "auto",
            ColorMode::Truecolor => "truecolor",
            ColorMode::Ansi256 => "ansi256",
            ColorMode::Mono => "mono",
        };
        f.write_str(s)
    }
}

impl std::str::FromStr for ColorMode {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("auto") {
            return Ok(ColorMode::Auto);
        }
        if s.eq_ignore_ascii_case("truecolor")
            || s.eq_ignore_ascii_case("true")
            || s.eq_ignore_ascii_case("24bit")
            || s.eq_ignore_ascii_case("24-bit")
        {
            return Ok(ColorMode::Truecolor);
        }
        if s.eq_ignore_ascii_case("ansi256")
            || s.eq_ignore_ascii_case("ansi-256")
            || s.eq_ignore_ascii_case("256")
            || s.eq_ignore_ascii_case("xterm256")
            || s.eq_ignore_ascii_case("xterm-256")
        {
            return Ok(ColorMode::Ansi256);
        }
        if s.eq_ignore_ascii_case("mono") || s.eq_ignore_ascii_case("monochrome") {
            return Ok(ColorMode::Mono);
        }
        Err(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TerminalPresenterConfig {
    pub color_mode: ColorMode,
}

impl Default for TerminalPresenterConfig {
    fn default() -> Self {
        Self {
            color_mode: ColorMode::Auto,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResolvedMode {
    Truecolor,
    Ansi256,
    Mono,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PrevCell {
    ch: char,
    fg: u32,
    bg: u32,
}

impl PrevCell {
    fn empty() -> Self {
        Self {
            ch: '\0',
            fg: 0,
            bg: 0,
        }
    }
}

pub struct TerminalPresenter {
    width: usize,
    height: usize,
    prev: Vec<PrevCell>,
    first: bool,
    config: TerminalPresenterConfig,
    prev_mode: ResolvedMode,
    cur_style: Option<(u32, u32)>,
}

impl TerminalPresenter {
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_config(width, height, TerminalPresenterConfig::default())
    }

    pub fn with_config(width: usize, height: usize, config: TerminalPresenterConfig) -> Self {
        let mut prev = Vec::new();
        prev.resize(width.saturating_mul(height), PrevCell::empty());
        Self {
            width,
            height,
            prev,
            first: true,
            config,
            prev_mode: ResolvedMode::Mono,
            cur_style: None,
        }
    }

    pub fn config(&self) -> &TerminalPresenterConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut TerminalPresenterConfig {
        &mut self.config
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        self.prev.clear();
        self.prev
            .resize(width.saturating_mul(height), PrevCell::empty());
        self.first = true;
        self.cur_style = None;
    }

    pub fn reset(&mut self) {
        self.prev.fill(PrevCell::empty());
        self.first = true;
        self.cur_style = None;
    }

    pub fn present<W: Write>(&mut self, out: &mut W, buf: &BufferTarget) -> io::Result<()> {
        let w = buf.width();
        let h = buf.height();

        if w != self.width || h != self.height {
            self.resize(w, h);
        }

        let mode = resolve_mode(self.config.color_mode);
        if mode != self.prev_mode {
            self.reset();
            self.prev_mode = mode;
        }

        if self.first {
            crossterm::queue!(
                out,
                terminal::Clear(terminal::ClearType::All),
                cursor::MoveTo(0, 0),
                style::ResetColor
            )?;
            self.prev.fill(PrevCell::empty());
            self.first = false;
        }

        let cells = buf.as_slice();
        let width = self.width;

        for y in 0..self.height {
            let row_start = y * width;
            let mut x = 0;
            while x < width {
                let idx = row_start + x;
                let key = make_prev_cell(&cells[idx], mode);
                if key == self.prev[idx] {
                    x += 1;
                    continue;
                }

                let run_start = x;
                let (fg_key, bg_key) = (key.fg, key.bg);

                let mut s = String::new();
                while x < width {
                    let idx2 = row_start + x;
                    let key2 = make_prev_cell(&cells[idx2], mode);
                    if key2 == self.prev[idx2] {
                        break;
                    }
                    if key2.fg != fg_key || key2.bg != bg_key {
                        break;
                    }
                    self.prev[idx2] = key2;
                    s.push(key2.ch);
                    x += 1;
                }

                let x0 = u16::try_from(run_start).unwrap_or(u16::MAX);
                let y0 = u16::try_from(y).unwrap_or(u16::MAX);
                crossterm::queue!(out, cursor::MoveTo(x0, y0))?;
                self.apply_style(out, mode, fg_key, bg_key)?;
                crossterm::queue!(out, style::Print(&s))?;
            }
        }

        out.flush()?;
        Ok(())
    }

    fn apply_style<W: Write>(
        &mut self,
        out: &mut W,
        mode: ResolvedMode,
        fg_key: u32,
        bg_key: u32,
    ) -> io::Result<()> {
        if mode == ResolvedMode::Mono {
            return Ok(());
        }
        if self.cur_style == Some((fg_key, bg_key)) {
            return Ok(());
        }
        match mode {
            ResolvedMode::Truecolor => {
                let fg = unpack_rgb(fg_key);
                let bg = unpack_rgb(bg_key);
                crossterm::queue!(
                    out,
                    style::SetForegroundColor(Color::Rgb {
                        r: fg.r,
                        g: fg.g,
                        b: fg.b
                    }),
                    style::SetBackgroundColor(Color::Rgb {
                        r: bg.r,
                        g: bg.g,
                        b: bg.b
                    })
                )?;
            }
            ResolvedMode::Ansi256 => {
                crossterm::queue!(
                    out,
                    style::SetForegroundColor(Color::AnsiValue(u8::try_from(fg_key).unwrap_or(0))),
                    style::SetBackgroundColor(Color::AnsiValue(u8::try_from(bg_key).unwrap_or(0)))
                )?;
            }
            ResolvedMode::Mono => {}
        }
        self.cur_style = Some((fg_key, bg_key));
        Ok(())
    }
}

pub struct TerminalTarget {
    presenter: TerminalPresenter,
    out: io::Stdout,
}

impl TerminalTarget {
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_config(width, height, TerminalPresenterConfig::default())
    }

    pub fn with_config(width: usize, height: usize, config: TerminalPresenterConfig) -> Self {
        Self {
            presenter: TerminalPresenter::with_config(width, height, config),
            out: io::stdout(),
        }
    }

    pub fn reset(&mut self) {
        self.presenter.reset();
    }

    pub fn present(&mut self, buf: &BufferTarget) -> io::Result<()> {
        self.presenter.present(&mut self.out, buf)
    }

    pub fn presenter_mut(&mut self) -> &mut TerminalPresenter {
        &mut self.presenter
    }
}

fn resolve_mode(mode: ColorMode) -> ResolvedMode {
    match mode {
        ColorMode::Truecolor => ResolvedMode::Truecolor,
        ColorMode::Ansi256 => ResolvedMode::Ansi256,
        ColorMode::Mono => ResolvedMode::Mono,
        ColorMode::Auto => detect_mode(),
    }
}

fn detect_mode() -> ResolvedMode {
    let colorterm = std::env::var("COLORTERM")
        .unwrap_or_default()
        .to_lowercase();
    if colorterm.contains("truecolor") || colorterm.contains("24bit") {
        return ResolvedMode::Truecolor;
    }
    let term = std::env::var("TERM").unwrap_or_default().to_lowercase();
    if term.contains("256color") {
        return ResolvedMode::Ansi256;
    }
    ResolvedMode::Mono
}

fn make_prev_cell(cell: &crate::targets::buffer::Cell, mode: ResolvedMode) -> PrevCell {
    match mode {
        ResolvedMode::Mono => PrevCell {
            ch: cell.ch,
            fg: 0,
            bg: 0,
        },
        ResolvedMode::Truecolor => PrevCell {
            ch: cell.ch,
            fg: pack_rgb(cell.fg),
            bg: pack_rgb(cell.bg),
        },
        ResolvedMode::Ansi256 => PrevCell {
            ch: cell.ch,
            fg: quantize_ansi256(cell.fg) as u32,
            bg: quantize_ansi256(cell.bg) as u32,
        },
    }
}

fn pack_rgb(c: Rgb8) -> u32 {
    (u32::from(c.r) << 16) | (u32::from(c.g) << 8) | u32::from(c.b)
}

fn unpack_rgb(v: u32) -> Rgb8 {
    Rgb8::new(
        ((v >> 16) & 255) as u8,
        ((v >> 8) & 255) as u8,
        (v & 255) as u8,
    )
}

fn quantize_ansi256(c: Rgb8) -> u8 {
    let (r, g, b) = (c.r, c.g, c.b);

    let gray = ((u16::from(r) + u16::from(g) + u16::from(b)) / 3) as u8;
    let gray_idx = gray_to_ansi256(gray);
    let gray_rgb = ansi256_to_rgb(gray_idx);

    let cube_idx = rgb_to_ansi256_cube(r, g, b);
    let cube_rgb = ansi256_to_rgb(cube_idx);

    let dg = dist2_rgb(r, g, b, gray_rgb.r, gray_rgb.g, gray_rgb.b);
    let dc = dist2_rgb(r, g, b, cube_rgb.r, cube_rgb.g, cube_rgb.b);

    if dg <= dc {
        gray_idx
    } else {
        cube_idx
    }
}

fn dist2_rgb(r0: u8, g0: u8, b0: u8, r1: u8, g1: u8, b1: u8) -> u32 {
    let dr = i32::from(r0) - i32::from(r1);
    let dg = i32::from(g0) - i32::from(g1);
    let db = i32::from(b0) - i32::from(b1);
    (dr * dr + dg * dg + db * db) as u32
}

fn rgb_to_ansi256_cube(r: u8, g: u8, b: u8) -> u8 {
    let ir = rgb_to_ansi6(r);
    let ig = rgb_to_ansi6(g);
    let ib = rgb_to_ansi6(b);
    (16u16 + 36u16 * u16::from(ir) + 6u16 * u16::from(ig) + u16::from(ib)) as u8
}

fn rgb_to_ansi6(v: u8) -> u8 {
    ((u16::from(v) * 5 + 127) / 255) as u8
}

fn gray_to_ansi256(gray: u8) -> u8 {
    if gray < 8 {
        return 16;
    }
    if gray > 248 {
        return 231;
    }
    (232u16 + ((u16::from(gray) - 8) / 10)) as u8
}

fn ansi256_to_rgb(idx: u8) -> Rgb8 {
    if idx < 16 {
        return ansi16_to_rgb(idx);
    }
    if idx >= 232 {
        let v = (8u16 + u16::from(idx - 232) * 10) as u8;
        return Rgb8::new(v, v, v);
    }
    let i = idx - 16;
    let r = i / 36;
    let g = (i % 36) / 6;
    let b = i % 6;
    Rgb8::new(ansi6_to_rgb(r), ansi6_to_rgb(g), ansi6_to_rgb(b))
}

fn ansi6_to_rgb(i: u8) -> u8 {
    match i {
        0 => 0,
        1 => 95,
        2 => 135,
        3 => 175,
        4 => 215,
        _ => 255,
    }
}

fn ansi16_to_rgb(idx: u8) -> Rgb8 {
    match idx {
        0 => Rgb8::new(0, 0, 0),
        1 => Rgb8::new(128, 0, 0),
        2 => Rgb8::new(0, 128, 0),
        3 => Rgb8::new(128, 128, 0),
        4 => Rgb8::new(0, 0, 128),
        5 => Rgb8::new(128, 0, 128),
        6 => Rgb8::new(0, 128, 128),
        7 => Rgb8::new(192, 192, 192),
        8 => Rgb8::new(128, 128, 128),
        9 => Rgb8::new(255, 0, 0),
        10 => Rgb8::new(0, 255, 0),
        11 => Rgb8::new(255, 255, 0),
        12 => Rgb8::new(0, 0, 255),
        13 => Rgb8::new(255, 0, 255),
        14 => Rgb8::new(0, 255, 255),
        _ => Rgb8::new(255, 255, 255),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::targets::buffer::Cell;

    #[test]
    fn quantize_basic_colors() {
        assert_eq!(quantize_ansi256(Rgb8::BLACK), 16);
        assert_eq!(quantize_ansi256(Rgb8::WHITE), 231);
        assert_eq!(quantize_ansi256(Rgb8::new(255, 0, 0)), 196);
        assert_eq!(quantize_ansi256(Rgb8::new(0, 255, 0)), 46);
        assert_eq!(quantize_ansi256(Rgb8::new(0, 0, 255)), 21);
    }

    #[test]
    fn color_mode_parse_and_display() {
        assert_eq!("auto".parse::<ColorMode>().unwrap(), ColorMode::Auto);
        assert_eq!(
            "TRUECOLOR".parse::<ColorMode>().unwrap(),
            ColorMode::Truecolor
        );
        assert_eq!("ansi-256".parse::<ColorMode>().unwrap(), ColorMode::Ansi256);
        assert_eq!("mono".parse::<ColorMode>().unwrap(), ColorMode::Mono);
        assert!("nope".parse::<ColorMode>().is_err());

        assert_eq!(ColorMode::Auto.to_string(), "auto");
        assert_eq!(ColorMode::Truecolor.to_string(), "truecolor");
        assert_eq!(ColorMode::Ansi256.to_string(), "ansi256");
        assert_eq!(ColorMode::Mono.to_string(), "mono");

        let cfg = TerminalPresenterConfig::default();
        assert_eq!(cfg.color_mode, ColorMode::Auto);
    }

    #[test]
    fn terminal_presenter_writes_diffs() {
        let mut buf = BufferTarget::new(4, 2);
        let mut p = TerminalPresenter::with_config(
            4,
            2,
            TerminalPresenterConfig {
                color_mode: ColorMode::Ansi256,
            },
        );

        buf.set(1, 0, Cell::new('A', Rgb8::new(255, 0, 0), Rgb8::BLACK, 0.0));
        buf.set(2, 0, Cell::new('B', Rgb8::new(0, 255, 0), Rgb8::BLACK, 0.0));

        let mut out = Vec::new();
        p.present(&mut out, &buf).unwrap();
        let s = String::from_utf8_lossy(&out);
        assert!(s.contains('A'));
        assert!(s.contains('B'));

        let mut out2 = Vec::new();
        p.present(&mut out2, &buf).unwrap();
        assert!(out2.is_empty());
    }
}
