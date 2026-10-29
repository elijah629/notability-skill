//! Terminal-only presentation; never changes training or puts animation on stdout.
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::{
    io::{self, IsTerminal, Write},
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    time::Duration,
};

#[derive(Debug)]
struct ForcedStderr;
fn move_cursor(n: usize, direction: char) -> io::Result<()> {
    if n == 0 {
        return Ok(());
    }
    write!(io::stderr(), "\x1b[{n}{direction}")
}
impl indicatif::TermLike for ForcedStderr {
    fn width(&self) -> u16 {
        std::env::var("COLUMNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100_u16)
            .clamp(20, 512)
    }
    fn move_cursor_up(&self, n: usize) -> io::Result<()> {
        move_cursor(n, 'A')
    }
    fn move_cursor_down(&self, n: usize) -> io::Result<()> {
        move_cursor(n, 'B')
    }
    fn move_cursor_right(&self, n: usize) -> io::Result<()> {
        move_cursor(n, 'C')
    }
    fn move_cursor_left(&self, n: usize) -> io::Result<()> {
        move_cursor(n, 'D')
    }
    fn write_line(&self, s: &str) -> io::Result<()> {
        writeln!(io::stderr(), "{s}")
    }
    fn write_str(&self, s: &str) -> io::Result<()> {
        write!(io::stderr(), "{s}")
    }
    fn clear_line(&self) -> io::Result<()> {
        write!(io::stderr(), "\r\x1b[2K")
    }
    fn flush(&self) -> io::Result<()> {
        io::stderr().flush()
    }
}

#[derive(Clone, Copy)]
pub enum Mode {
    Auto,
    Always,
    Never,
}
static MODE: AtomicU8 = AtomicU8::new(0);
static QUIET: AtomicBool = AtomicBool::new(false);
pub fn configure(mode: Mode, quiet: bool) {
    MODE.store(
        match mode {
            Mode::Auto => 0,
            Mode::Always => 1,
            Mode::Never => 2,
        },
        Ordering::Relaxed,
    );
    QUIET.store(quiet, Ordering::Relaxed);
}
fn visible() -> bool {
    !QUIET.load(Ordering::Relaxed)
        && match MODE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                std::io::stderr().is_terminal() && std::env::var("TERM").is_ok_and(|s| s != "dumb")
            }
        }
}
pub fn log(message: impl AsRef<str>) {
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("{}", message.as_ref());
    }
}
pub struct Progress {
    bar: ProgressBar,
    done: bool,
    label: String,
}
impl Progress {
    fn new(label: &str, length: Option<u64>, template: &str) -> Self {
        let target = if !visible() {
            ProgressDrawTarget::hidden()
        } else if MODE.load(Ordering::Relaxed) == 1 {
            ProgressDrawTarget::term_like_with_hz(Box::new(ForcedStderr), 10)
        } else {
            ProgressDrawTarget::stderr_with_hz(10)
        };
        let bar = ProgressBar::with_draw_target(length, target);
        bar.set_style(
            ProgressStyle::with_template(template)
                .expect("static progress template")
                .progress_chars("=>-"),
        );
        bar.set_prefix(label.to_owned());
        if !bar.is_hidden() {
            bar.enable_steady_tick(Duration::from_millis(100));
        }
        Self {
            bar,
            done: false,
            label: label.into(),
        }
    }
    pub fn count(label: &str, total: usize) -> Self {
        Self::new(
            label,
            Some(total as u64),
            "{spinner:.cyan} {prefix:.bold} [{bar:24.cyan/blue}] {pos}/{len} {percent}%  {elapsed_precise}  ETA {eta_precise}  {msg}",
        )
    }
    pub fn timed(label: &str, seconds: f64) -> Self {
        Self::new(
            label,
            Some((seconds.max(0.) * 1000.).ceil() as u64),
            "{spinner:.cyan} {prefix:.bold} [{bar:24.cyan/blue}] {percent}%  {elapsed_precise}  remaining {eta_precise}  {msg}",
        )
    }
    #[cfg(any(feature = "rocm", test))]
    fn epoch(label: &str, total: usize, completed: usize) -> Self {
        let progress = Self::new(
            label,
            Some(total as u64),
            "{spinner:.cyan} {prefix:.bold} [{wide_bar:.cyan/blue}] {pos}/{len} batches  {elapsed_precise}  ETA {eta_precise}  {msg}",
        );
        progress.set(completed);
        progress
    }
    #[cfg(any(feature = "rocm", test))]
    fn epoch_summary(&self) {
        self.bar.set_style(
            ProgressStyle::with_template(
                "{prefix:.bold}  {pos}/{len} batches  {elapsed_precise}  {msg}",
            )
            .expect("static epoch summary template"),
        );
    }
    #[cfg(any(feature = "rocm", test))]
    fn pause(mut self, message: &str) {
        let visible = !self.bar.is_hidden();
        self.done = true;
        self.epoch_summary();
        // finish_with_message advances to the total, which would falsely show
        // a complete epoch when the time budget stopped between batches.
        self.bar.abandon_with_message(message.to_owned());
        if visible {
            eprintln!();
        }
    }
    pub fn spinner(label: &str) -> Self {
        Self::new(
            label,
            None,
            "{spinner:.cyan} {prefix:.bold}  {elapsed_precise}  {msg}",
        )
    }
    pub fn set(&self, position: usize) {
        self.bar.set_position(position as u64);
    }
    pub fn update_time(&self, elapsed: Duration, message: impl Into<String>) {
        self.bar
            .set_position((elapsed.as_millis() as u64).min(self.bar.length().unwrap_or(u64::MAX)));
        self.bar.set_message(message.into());
    }
    pub fn message(&self, message: impl Into<String>) {
        self.bar.set_message(message.into());
    }
    pub fn note(&self, message: impl AsRef<str>) {
        if self.bar.is_hidden() {
            log(message);
        } else {
            self.bar.println(message.as_ref());
        }
    }
    pub fn suspend<T>(&self, f: impl FnOnce() -> T) -> T {
        self.bar.suspend(f)
    }
    pub fn finish(mut self, message: impl Into<String>) {
        let visible = !self.bar.is_hidden();
        self.done = true;
        self.bar.finish_with_message(message.into());
        if visible {
            eprintln!();
        }
    }
}

/// One retained line per completed epoch. Batch updates only redraw that line;
/// loss is a sample-weighted mean and is never displayed for a partial epoch.
#[cfg(any(feature = "rocm", test))]
pub(crate) struct EpochProgress {
    label: String,
    batches: usize,
    step: u64,
    bar: Option<Progress>,
    loss_sum: f64,
    samples: usize,
    complete_observation: bool,
}

#[cfg(any(feature = "rocm", test))]
pub(crate) struct EpochSummary {
    pub epoch: u64,
    pub mean_loss: Option<f64>,
}
#[cfg(any(feature = "rocm", test))]
impl EpochSummary {
    fn message(&self) -> String {
        match self.mean_loss {
            Some(loss) => format!("complete · mean loss {loss:.6}"),
            None => "complete · resumed; loss unavailable".into(),
        }
    }
}

#[cfg(any(feature = "rocm", test))]
impl EpochProgress {
    pub fn new(label: &str, batches: usize, step: u64) -> Self {
        assert!(batches > 0);
        Self {
            label: label.into(),
            batches,
            step,
            bar: None,
            loss_sum: 0.,
            samples: 0,
            complete_observation: step.is_multiple_of(batches as u64),
        }
    }
    fn ensure_bar(&mut self) -> &Progress {
        self.bar.get_or_insert_with(|| {
            Progress::epoch(
                &format!(
                    "{} · Epoch {}",
                    self.label,
                    self.step / self.batches as u64 + 1
                ),
                self.batches,
                self.step as usize % self.batches,
            )
        })
    }
    pub fn start(&mut self) {
        self.ensure_bar();
    }
    pub fn update(&mut self, loss: f64, samples: usize) -> Option<EpochSummary> {
        self.ensure_bar();
        self.step += 1;
        self.loss_sum += loss * samples as f64;
        self.samples += samples;
        let position = self.step as usize % self.batches;
        if position != 0 {
            let bar = self.bar.as_ref().unwrap();
            bar.set(position);
            bar.message("");
            return None;
        }
        let summary = EpochSummary {
            epoch: self.step / self.batches as u64,
            mean_loss: self
                .complete_observation
                .then(|| self.loss_sum / self.samples as f64),
        };
        let bar = self.bar.take().unwrap();
        bar.set(self.batches);
        bar.epoch_summary();
        let message = summary.message();
        if bar.bar.is_hidden() {
            log(format!(
                "{} · Epoch {} · {message}",
                self.label, summary.epoch
            ));
        }
        bar.finish(message);
        self.loss_sum = 0.;
        self.samples = 0;
        self.complete_observation = true;
        Some(summary)
    }
    pub fn message(&mut self, message: &str) {
        if let Some(bar) = &self.bar {
            bar.message(message);
        }
    }
    pub fn finish(mut self) {
        if let Some(bar) = self.bar.take() {
            let message = format!(
                "{} · Epoch {} paused · {}/{} batches · checkpoint saved",
                self.label,
                self.step / self.batches as u64 + 1,
                self.step as usize % self.batches,
                self.batches
            );
            if bar.bar.is_hidden() {
                log(&message);
            }
            bar.pause("paused · checkpoint saved");
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        if !self.done {
            let visible = !self.bar.is_hidden();
            self.bar
                .abandon_with_message(format!("{} stopped", self.label));
            if visible {
                eprintln!();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_loss_waits_for_all_batches_and_weights_short_batch() {
        let mut progress = EpochProgress::new("Sequence", 3, 0);
        progress.start();
        progress.message("Saving checkpoint…");
        assert!(progress.update(1., 8).is_none());
        assert!(progress.update(2., 8).is_none());
        let summary = progress.update(3., 2).unwrap();
        assert_eq!(summary.epoch, 1);
        assert_eq!(summary.mean_loss, Some(30. / 18.));
        assert_eq!(summary.message(), "complete · mean loss 1.666667");
        assert!(progress.update(100., 8).is_none());
        let paused = progress.bar.as_ref().unwrap().bar.clone();
        progress.finish(); // A budget stop must not produce an epoch-loss summary.
        assert_eq!(paused.position(), 1);
        assert_eq!(paused.length(), Some(3));
    }

    #[test]
    fn resumed_partial_epoch_does_not_mislabel_partial_loss() {
        let mut progress = EpochProgress::new("Sequence", 2, 3);
        let resumed = progress.update(100., 8).unwrap();
        assert_eq!(resumed.epoch, 2);
        assert_eq!(resumed.mean_loss, None);
        assert!(!resumed.message().contains("mean loss"));
        assert!(progress.update(2., 8).is_none());
        let complete = progress.update(4., 2).unwrap();
        assert_eq!(complete.epoch, 3);
        assert_eq!(complete.mean_loss, Some(2.4));
        assert!(progress.bar.is_none());
        progress.finish(); // No empty next epoch after an exact boundary.
    }
}
