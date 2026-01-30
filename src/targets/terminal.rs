use crate::targets::buffer::BufferTarget;
use crossterm::{
    cursor,
    style,
    terminal,
};
use std::io::{
    self,
    Write,
};

pub struct TerminalGuard {
    active: bool,
}

impl TerminalGuard {
    pub fn new() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        crossterm::execute!(
            io::stdout(),
            terminal::EnterAlternateScreen,
            cursor::Hide,
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        )?;
        Ok(Self { active: true })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let _ = crossterm::execute!(
            io::stdout(),
            cursor::Show,
            terminal::LeaveAlternateScreen
        );
        let _ = terminal::disable_raw_mode();
        self.active = false;
    }
}

pub struct TerminalPresenter {
    width: usize,
    height: usize,
    prev: Vec<char>,
    first: bool,
}

impl TerminalPresenter {
    pub fn new(width: usize, height: usize) -> Self {
        let mut prev = Vec::new();
        prev.resize(width.saturating_mul(height), '\0');
        Self {
            width,
            height,
            prev,
            first: true,
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        self.prev.clear();
        self.prev.resize(width.saturating_mul(height), '\0');
        self.first = true;
    }

    pub fn reset(&mut self) {
        self.prev.fill('\0');
        self.first = true;
    }

    pub fn present<W: Write>(&mut self, out: &mut W, buf: &BufferTarget) -> io::Result<()> {
        let w = buf.width();
        let h = buf.height();

        if w != self.width || h != self.height {
            self.resize(w, h);
        }

        if self.first {
            crossterm::queue!(
                out,
                terminal::Clear(terminal::ClearType::All),
                cursor::MoveTo(0, 0)
            )?;
            self.prev.fill('\0');
            self.first = false;
        }

        let cells = buf.as_slice();
        let width = self.width;

        for y in 0..self.height {
            let row_start = y * width;
            let mut x = 0;
            while x < width {
                let idx = row_start + x;
                let ch = cells[idx].ch;
                if ch == self.prev[idx] {
                    x += 1;
                    continue;
                }

                let run_start = x;
                let mut s = String::new();
                while x < width {
                    let idx2 = row_start + x;
                    let ch2 = cells[idx2].ch;
                    if ch2 == self.prev[idx2] {
                        break;
                    }
                    self.prev[idx2] = ch2;
                    s.push(ch2);
                    x += 1;
                }

                crossterm::queue!(
                    out,
                    cursor::MoveTo(run_start as u16, y as u16),
                    style::Print(&s)
                )?;
            }
        }

        out.flush()?;
        Ok(())
    }
}

pub struct TerminalTarget {
    presenter: TerminalPresenter,
    out: io::Stdout,
}

impl TerminalTarget {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            presenter: TerminalPresenter::new(width, height),
            out: io::stdout(),
        }
    }

    pub fn reset(&mut self) {
        self.presenter.reset();
    }

    pub fn present(&mut self, buf: &BufferTarget) -> io::Result<()> {
        self.presenter.present(&mut self.out, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::targets::buffer::Cell;

    #[test]
    fn terminal_presenter_writes_diffs() {
        let mut buf = BufferTarget::new(4, 2);
        let mut p = TerminalPresenter::new(4, 2);

        buf.set(1, 0, Cell::new('A', 0, 0, 0.0));
        buf.set(2, 0, Cell::new('B', 0, 0, 0.0));

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
