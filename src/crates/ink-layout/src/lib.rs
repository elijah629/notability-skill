use anyhow::{Context, Result, bail, ensure};
use ink_corpus::{Stroke, glyph_label};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    Glyph(String),
    Space,
    Row(Vec<Ast>),
    Fraction(Box<Ast>, Box<Ast>),
    Root(Box<Ast>),
    IndexedRoot(Box<Ast>, Box<Ast>),
    Enclosure(String, Box<Ast>),
    Delimited(String, String, Box<Ast>),
    Phantom(Box<Ast>),
    Accent(String, Box<Ast>),
    Table(String, Vec<Vec<Ast>>),
    Scripts {
        base: Box<Ast>,
        sup: Option<Box<Ast>>,
        sub: Option<Box<Ast>>,
    },
}
struct Parser {
    chars: Vec<char>,
    p: usize,
    depth: usize,
}
pub fn parse(input: &str) -> Result<Ast> {
    ensure!(input.len() <= 16384, "input too long");
    ensure!(input.matches("\\left").count() <= 64, "LaTeX nesting limit");
    let input = input.trim();
    let input = if input.starts_with('$') && input.ends_with('$') && input.len() >= 2 {
        &input[1..input.len() - 1]
    } else {
        input
    };
    let mut p = Parser {
        chars: input.chars().collect(),
        p: 0,
        depth: 0,
    };
    let ast = p.row(false, false)?;
    ensure!(p.p == p.chars.len(), "trailing input");
    Ok(ast)
}
impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.p).copied()
    }
    fn starts(&self, s: &str) -> bool {
        self.chars[self.p..]
            .iter()
            .copied()
            .zip(s.chars())
            .filter(|(a, b)| a == b)
            .count()
            == s.chars().count()
            && self.chars[self.p..].len() >= s.chars().count()
    }
    fn spaces(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.p += 1;
        }
    }
    fn raw_group(&mut self) -> Result<String> {
        self.spaces();
        ensure!(self.peek() == Some('{'), "expected braced group");
        self.p += 1;
        let start = self.p;
        let mut level = 1;
        while let Some(c) = self.peek() {
            self.p += 1;
            if c == '\\' {
                ensure!(self.peek().is_some(), "unfinished escape");
                self.p += 1;
                continue;
            }
            if c == '{' {
                level += 1;
            }
            if c == '}' {
                level -= 1;
                if level == 0 {
                    return Ok(self.chars[start..self.p - 1].iter().collect());
                }
            }
        }
        bail!("unclosed group")
    }
    fn group(&mut self) -> Result<Ast> {
        self.spaces();
        ensure!(self.peek() == Some('{'), "expected braced group");
        self.p += 1;
        self.row(true, false)
    }
    fn argument(&mut self) -> Result<Ast> {
        self.spaces();
        if self.peek() == Some('{') {
            self.group()
        } else {
            self.atom()
        }
    }
    fn row(&mut self, braced: bool, cell: bool) -> Result<Ast> {
        self.depth += 1;
        ensure!(self.depth <= 64, "LaTeX nesting limit");
        let mut nodes = Vec::new();
        while let Some(c) = self.peek() {
            if cell && (c == '&' || self.starts("\\\\") || self.starts("\\end")) {
                break;
            }
            if c == '}' {
                ensure!(braced, "unmatched closing brace");
                self.p += 1;
                self.depth -= 1;
                return Ok(Ast::Row(nodes));
            }
            let mut base = self.atom()?;
            let (mut sup, mut sub) = (None, None);
            while matches!(self.peek(), Some('^' | '_')) {
                let op = self.peek().unwrap();
                self.p += 1;
                let value = Box::new(self.argument()?);
                if op == '^' {
                    ensure!(sup.is_none(), "duplicate superscript");
                    sup = Some(value);
                } else {
                    ensure!(sub.is_none(), "duplicate subscript");
                    sub = Some(value);
                }
            }
            if sup.is_some() || sub.is_some() {
                base = Ast::Scripts {
                    base: Box::new(base),
                    sup,
                    sub,
                };
            }
            nodes.push(base);
        }
        ensure!(!braced, "unclosed group");
        self.depth -= 1;
        Ok(Ast::Row(nodes))
    }
    fn table(&mut self) -> Result<Ast> {
        let name = self.raw_group()?;
        ensure!(
            [
                "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "cases",
                "aligned", "gathered", "array"
            ]
            .contains(&name.as_str()),
            "unsupported environment {name}"
        );
        if name == "array" {
            let format = self.raw_group()?;
            ensure!(
                format.chars().all(|c| "lcr| ".contains(c)),
                "unsupported array column format"
            );
        }
        let mut rows = vec![];
        let mut row = vec![];
        loop {
            ensure!(self.peek().is_some(), "unclosed environment {name}");
            row.push(self.row(false, true)?);
            if self.starts("\\end") {
                self.p += 4;
                ensure!(self.raw_group()? == name, "mismatched environment end");
                rows.push(row);
                break;
            }
            if self.peek() == Some('&') {
                self.p += 1;
            } else {
                ensure!(self.starts("\\\\"), "expected table separator");
                self.p += 2;
                rows.push(row);
                row = vec![];
            }
            ensure!(rows.len() < 256 && row.len() < 128, "table size limit");
        }
        Ok(Ast::Table(name, rows))
    }
    fn delimiter(&mut self) -> Result<String> {
        self.spaces();
        let c = self.peek().context("missing delimiter")?;
        self.p += 1;
        if c == '\\' {
            let start = self.p;
            while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                self.p += 1;
            }
            if self.p == start {
                let c = self.peek().context("missing delimiter")?;
                self.p += 1;
                return Ok(c.to_string());
            }
            let name: String = self.chars[start..self.p].iter().collect();
            return match name.as_str() {
                "lbrace" => Ok("{".into()),
                "rbrace" => Ok("}".into()),
                "langle" => Ok("<".into()),
                "rangle" => Ok(">".into()),
                "lvert" | "rvert" | "vert" => Ok("|".into()),
                "Vert" => Ok("||".into()),
                _ => bail!("unsupported delimiter {name}"),
            };
        }
        ensure!("()[]{}|<>.".contains(c), "invalid delimiter");
        Ok(c.to_string())
    }
    fn delimited(&mut self) -> Result<Ast> {
        let left = self.delimiter()?;
        let start = self.p;
        let mut level = 1;
        while self.peek().is_some() {
            if self.starts("\\left") {
                level += 1;
                self.p += 5;
            } else if self.starts("\\right") {
                level -= 1;
                if level == 0 {
                    let body: String = self.chars[start..self.p].iter().collect();
                    self.p += 6;
                    let right = self.delimiter()?;
                    return Ok(Ast::Delimited(left, right, Box::new(parse(&body)?)));
                }
                self.p += 6;
            } else {
                self.p += 1;
            }
        }
        bail!("missing \\right")
    }
    fn atom(&mut self) -> Result<Ast> {
        let c = self.peek().context("missing script/argument")?;
        self.p += 1;
        match c {
            '{' => self.row(true, false),
            '\\' => {
                if self.peek().is_some_and(|c| !c.is_ascii_alphabetic()) {
                    let c = self.peek().context("unfinished escape")?;
                    self.p += 1;
                    return match c {
                        ' ' | ',' | ';' | ':' | '!' => Ok(Ast::Space),
                        '%' | '$' | '#' | '&' | '_' | '{' | '}' | '\\' => {
                            Ok(Ast::Glyph(c.to_string()))
                        }
                        _ => bail!("unsupported LaTeX escape \\{c}"),
                    };
                }
                let start = self.p;
                while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                    self.p += 1;
                }
                ensure!(self.p > start, "unfinished command");
                let name: String = self.chars[start..self.p].iter().collect();
                match name.as_str() {
                    "frac" | "dfrac" | "tfrac" => Ok(Ast::Fraction(
                        Box::new(self.argument()?),
                        Box::new(self.argument()?),
                    )),
                    "sqrt" => {
                        self.spaces();
                        let index = if self.peek() == Some('[') {
                            self.p += 1;
                            let start = self.p;
                            while self.peek().is_some_and(|c| c != ']') {
                                self.p += 1;
                            }
                            ensure!(self.peek() == Some(']'), "unclosed root index");
                            let s: String = self.chars[start..self.p].iter().collect();
                            self.p += 1;
                            Some(parse(&s)?)
                        } else {
                            None
                        };
                        let body = Box::new(self.argument()?);
                        Ok(if let Some(i) = index {
                            Ast::IndexedRoot(Box::new(i), body)
                        } else {
                            Ast::Root(body)
                        })
                    }
                    "parallel" => Ok(Ast::Row(vec![
                        Ast::Glyph("/".into()),
                        Ast::Glyph("/".into()),
                    ])),
                    "boxed" => Ok(Ast::Enclosure("box".into(), Box::new(self.group()?))),
                    "enclose" => {
                        let role = self.raw_group()?;
                        ensure!(role == "circle", "unsupported enclosure {role}");
                        Ok(Ast::Enclosure(role, Box::new(self.group()?)))
                    }
                    "phantom" => Ok(Ast::Phantom(Box::new(self.group()?))),
                    "vec" | "overline" | "hat" => Ok(Ast::Accent(name, Box::new(self.argument()?))),
                    "text" | "operatorname" => {
                        let text = self.raw_group()?;
                        let mut v = vec![];
                        let mut chars = text.chars();
                        while let Some(c) = chars.next() {
                            let c = if c == '\\' {
                                let escaped = chars.next().context("unfinished text escape")?;
                                ensure!("%$#&_{}\\ ".contains(escaped), "unsupported text escape");
                                escaped
                            } else {
                                c
                            };
                            ensure!(
                                !c.is_control() || c.is_whitespace(),
                                "control character in text"
                            );
                            v.push(if c.is_whitespace() {
                                Ast::Space
                            } else {
                                Ast::Glyph(c.to_string())
                            });
                        }
                        Ok(Ast::Row(v))
                    }
                    "mathrm" | "mathbb" | "mathbf" | "mathit" => self.argument(),
                    "begin" => self.table(),
                    "quad" | "qquad" => Ok(Ast::Row(vec![
                        Ast::Space;
                        if name == "quad" { 2 } else { 4 }
                    ])),
                    "left" => self.delimited(),
                    "sin" | "cos" | "tan" | "log" | "ln" | "exp" | "lim" => Ok(Ast::Row(
                        name.chars().map(|c| Ast::Glyph(c.to_string())).collect(),
                    )),
                    _ => {
                        let name = match name.as_str() {
                            "le" => "leq",
                            "ge" => "geq",
                            "ne" => "neq",
                            "to" => "rightarrow",
                            "lvert" | "rvert" => return Ok(Ast::Glyph("|".into())),
                            "lbrace" => return Ok(Ast::Glyph("{".into())),
                            "rbrace" => return Ok(Ast::Glyph("}".into())),
                            _ => &name,
                        };
                        let label = format!("\\{name}");
                        ensure!(glyph_label(&label), "unsupported command {label}");
                        Ok(Ast::Glyph(label))
                    }
                }
            }
            c if c.is_whitespace() => Ok(Ast::Space),
            '^' | '_' | '&' | '$' => bail!("unsupported syntax {c:?}"),
            c if !c.is_control() => Ok(Ast::Glyph(c.to_string())),
            _ => bail!("unsupported syntax {c:?}"),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Glyph {
    /// Conditions for a jointly decoded run; strokes are never split into letters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub run_labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_label: Option<String>,
    pub label: String,
    pub mode: String,
    pub model_id: String,
    pub decoder_sha256: String,
    pub latent_seed: u64,
    pub strokes: Vec<Stroke>,
    pub transform: [f32; 4],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Procedure {
    pub role: String,
    pub mode: String,
    pub points: Vec<[f32; 2]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Drawing {
    pub schema_version: u32,
    pub input: String,
    pub seed: u64,
    pub width: f32,
    pub height: f32,
    /// Output-space pen width. None retains supplied widths for source previews.
    #[serde(default)]
    pub pen_width: Option<f32>,
    pub glyphs: Vec<Glyph>,
    pub procedures: Vec<Procedure>,
}
pub const PEN_WIDTH: f32 = 0.025;
pub fn procedural_symbol(label: &str) -> bool {
    let _ = label;
    false
}
pub fn neural_condition(label: &str) -> &str {
    label
}

pub fn delimiter_labels(delimiter: &str) -> Vec<String> {
    match delimiter {
        "." => vec![],
        "||" => vec!["|".into(), "|".into()],
        label => vec![label.into()],
    }
}
pub fn table_delimiters(kind: &str) -> (&'static str, &'static str) {
    match kind {
        "bmatrix" => ("[", "]"),
        "pmatrix" => ("(", ")"),
        "Bmatrix" => ("{", "}"),
        "vmatrix" => ("|", "|"),
        "Vmatrix" => ("||", "||"),
        "cases" => ("{", "."),
        _ => (".", "."),
    }
}
pub fn drawing_labels(drawing: &Drawing) -> Vec<String> {
    drawing
        .glyphs
        .iter()
        .flat_map(|g| {
            if g.run_labels.is_empty() && !g.mode.starts_with("neural-run") {
                vec![g.label.clone()]
            } else {
                g.run_labels
                    .iter()
                    .filter(|s| s.as_str() != "<space>")
                    .cloned()
                    .collect()
            }
        })
        .collect()
}

/// Joint generation preserves pen lifts and cross-letter strokes. None opts out;
/// decoder errors must propagate, never fall back to artificial connecting lines.
pub trait InkProvider {
    fn glyph(&mut self, label: &str, seed: u64) -> Result<Glyph>;
    fn run(&mut self, _labels: &[String], _seed: u64) -> Result<Option<Glyph>> {
        Ok(None)
    }
}
impl<F: FnMut(&str, u64) -> Result<Glyph>> InkProvider for F {
    fn glyph(&mut self, label: &str, seed: u64) -> Result<Glyph> {
        self(label, seed)
    }
}
struct Boxed {
    w: f32,
    above: f32,
    below: f32,
    glyphs: Vec<Glyph>,
    procedures: Vec<Procedure>,
}
impl Boxed {
    fn empty() -> Self {
        Self {
            w: 0.,
            above: 0.,
            below: 0.,
            glyphs: vec![],
            procedures: vec![],
        }
    }
    fn move_by(&mut self, x: f32, y: f32) {
        for g in &mut self.glyphs {
            g.transform[2] += x;
            g.transform[3] += y;
        }
        for p in &mut self.procedures {
            for q in &mut p.points {
                q[0] += x;
                q[1] += y;
            }
        }
    }
    fn scale(&mut self, s: f32) {
        self.w *= s;
        self.above *= s;
        self.below *= s;
        for g in &mut self.glyphs {
            for v in &mut g.transform {
                *v *= s;
            }
        }
        for p in &mut self.procedures {
            for q in &mut p.points {
                q[0] *= s;
                q[1] *= s;
            }
        }
    }
    fn append(&mut self, other: Self) {
        self.glyphs.extend(other.glyphs);
        self.procedures.extend(other.procedures);
    }
}
struct Layout<'a> {
    provider: &'a mut dyn InkProvider,
    rng: u64,
}
impl Layout<'_> {
    fn run_box(mut glyph: Glyph) -> Result<Boxed> {
        ensure!(!glyph.strokes.is_empty(), "run decoder returned no strokes");
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for p in glyph.strokes.iter().flat_map(|s| &s.points) {
            ensure!(p.iter().all(|v| v.is_finite()), "nonfinite run");
            bounds[0] = bounds[0].min(p[0]);
            bounds[1] = bounds[1].min(p[1]);
            bounds[2] = bounds[2].max(p[0]);
            bounds[3] = bounds[3].max(p[1]);
        }
        let metrics: Vec<_> = glyph
            .run_labels
            .iter()
            .filter(|s| s.as_str() != "<space>")
            .map(|s| font_metrics(s))
            .collect();
        let above = if metrics.is_empty() {
            1.
        } else {
            metrics.iter().map(|m| m.0).fold(0., f32::max)
        };
        let below = metrics.iter().map(|m| m.1).fold(0., f32::max);
        let scale = (above + below) / (bounds[3] - bounds[1]).max(1e-6);
        // A single isotropic transform retains learned drift, slant, spacing and joins.
        glyph.transform = [scale, scale, -bounds[0] * scale, -bounds[3] * scale + below];
        Ok(Boxed {
            w: (bounds[2] - bounds[0]) * scale + 0.15,
            above,
            below,
            glyphs: vec![glyph],
            procedures: vec![],
        })
    }
    fn enclose(&mut self, mut inner: Boxed, left: &str, right: &str) -> Result<Boxed> {
        let top = -inner.above - 0.1;
        let bottom = inner.below + 0.1;
        let mut opening = self.delimiter(left, top, bottom)?;
        let mut closing = self.delimiter(right, top, bottom)?;
        let width = opening.w + inner.w + closing.w;
        inner.move_by(opening.w, 0.);
        closing.move_by(width - closing.w, 0.);
        opening.append(inner);
        opening.append(closing);
        opening.w = width;
        opening.above = (-top).max(0.);
        opening.below = bottom.max(0.);
        Ok(opening)
    }
    fn delimiter(&mut self, label: &str, top: f32, bottom: f32) -> Result<Boxed> {
        let mut out = Boxed::empty();
        for label in delimiter_labels(label) {
            let mut b = self.layout(&Ast::Glyph(label))?;
            let factor = (bottom - top) / (b.above + b.below).max(0.05);
            let glyph = &mut b.glyphs[0];
            glyph.transform[1] *= factor;
            glyph.transform[3] = top + (glyph.transform[3] + b.above) * factor;
            b.move_by(out.w, 0.);
            out.w += b.w;
            out.append(b);
        }
        out.above = (-top).max(0.);
        out.below = bottom.max(0.);
        Ok(out)
    }
    fn layout(&mut self, ast: &Ast) -> Result<Boxed> {
        match ast {
            Ast::Space => Ok(Boxed {
                w: 0.4,
                ..Boxed::empty()
            }),
            Ast::Glyph(label) => {
                self.rng = self.rng.wrapping_add(0x9e3779b97f4a7c15);
                let mut glyph = self.provider.glyph(label, self.rng)?;
                ensure!(
                    !glyph.strokes.is_empty(),
                    "neural decoder returned no strokes"
                );
                let mut b = [
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                ];
                for p in glyph.strokes.iter().flat_map(|s| &s.points) {
                    ensure!(p.iter().all(|v| v.is_finite()), "nonfinite glyph");
                    b[0] = b[0].min(p[0]);
                    b[1] = b[1].min(p[1]);
                    b[2] = b[2].max(p[0]);
                    b[3] = b[3].max(p[1]);
                }
                let operators = [
                    "+", "-", "=", "*", "\\times", "\\div", "\\pm", "\\leq", "\\geq", "\\neq",
                    "\\approx", "\\sim", "\\propto", "\\in", "\\circ",
                ];
                if operators.contains(&label.as_str()) {
                    let size = if label == "\\circ" { 0.25 } else { 0.8 };
                    let scale = size / (b[2] - b[0]).max(b[3] - b[1]).max(0.05);
                    let height = (b[3] - b[1]) * scale;
                    glyph.transform = [
                        scale,
                        scale,
                        -b[0] * scale,
                        -(b[1] + b[3]) * 0.5 * scale - 0.4,
                    ];
                    return Ok(Boxed {
                        w: (b[2] - b[0]) * scale + 0.15,
                        above: 0.4 + height * 0.5,
                        below: (height * 0.5 - 0.4).max(0.),
                        glyphs: vec![glyph],
                        procedures: vec![],
                    });
                }
                let h = (b[3] - b[1]).max(0.05);
                let (above, descender) = font_metrics(label);
                let height = above + descender;
                let w = (b[2] - b[0]).max(0.01) / h * height;
                glyph.transform = [
                    height / h,
                    height / h,
                    -b[0] * height / h,
                    -b[3] * height / h + descender,
                ];
                Ok(Boxed {
                    w: w + 0.15,
                    above: height - descender,
                    below: descender,
                    glyphs: vec![glyph],
                    procedures: vec![],
                })
            }
            Ast::Row(nodes) => {
                let mut result = Boxed::empty();
                let mut index = 0;
                while index < nodes.len() {
                    let start = index;
                    let mut conditions = vec![];
                    while index < nodes.len() {
                        match &nodes[index] {
                            Ast::Glyph(label) => conditions.push(label.clone()),
                            Ast::Space => conditions.push("<space>".into()),
                            _ => break,
                        }
                        index += 1;
                    }
                    let run = if conditions
                        .iter()
                        .filter(|s| s.as_str() != "<space>")
                        .count()
                        > 1
                    {
                        self.rng = self.rng.wrapping_add(0x9e3779b97f4a7c15);
                        self.provider.run(&conditions, self.rng)?
                    } else {
                        None
                    };
                    if let Some(glyph) = run {
                        ensure!(
                            glyph.run_labels == conditions,
                            "run decoder changed requested conditions"
                        );
                        let mut b = Self::run_box(glyph)?;
                        b.move_by(result.w, 0.);
                        result.w += b.w;
                        result.above = result.above.max(b.above);
                        result.below = result.below.max(b.below);
                        result.append(b);
                    } else {
                        let end = index.max(start + 1);
                        for node in &nodes[start..end] {
                            let mut b = self.layout(node)?;
                            b.move_by(result.w, 0.);
                            result.w += b.w;
                            result.above = result.above.max(b.above);
                            result.below = result.below.max(b.below);
                            result.append(b);
                        }
                        index = end;
                    }
                }
                Ok(result)
            }
            Ast::Fraction(top, bottom) => {
                let mut a = self.layout(top)?;
                let mut b = self.layout(bottom)?;
                a.scale(0.8);
                b.scale(0.8);
                let w = a.w.max(b.w) + 0.4;
                let ay = -0.35 - a.below;
                let by = 0.25 + b.above;
                a.move_by((w - a.w) / 2., ay);
                b.move_by((w - b.w) / 2., by);
                let mut result = Boxed {
                    w,
                    above: a.above - ay,
                    below: b.below + by,
                    ..Boxed::empty()
                };
                result.append(a);
                result.append(b);
                result.procedures.push(Procedure {
                    role: "fraction bar".into(),
                    mode: "procedural".into(),
                    points: vec![[0., -0.05], [w, -0.05]],
                });
                Ok(result)
            }
            Ast::Root(inner) => {
                let mut a = self.layout(inner)?;
                a.move_by(0.6, 0.);
                let w = a.w + 0.7;
                let above = a.above + 0.15;
                let below = a.below;
                let mut result = Boxed {
                    w,
                    above,
                    below,
                    ..Boxed::empty()
                };
                result.append(a);
                result.procedures.push(Procedure {
                    role: "radical".into(),
                    mode: "procedural".into(),
                    points: vec![
                        [0., -0.3],
                        [0.15, -0.4],
                        [0.3, 0.05],
                        [0.55, -above],
                        [w, -above],
                    ],
                });
                Ok(result)
            }
            Ast::Delimited(left, right, inner) => {
                let b = self.layout(inner)?;
                self.enclose(b, left, right)
            }
            Ast::IndexedRoot(index, inner) => {
                let mut root = self.layout(&Ast::Root(inner.clone()))?;
                let mut index = self.layout(index)?;
                index.scale(0.4);
                index.move_by(0., -root.above + index.above);
                root.above = root.above.max(index.above + root.above * 0.6);
                let mut glyphs = index.glyphs;
                glyphs.append(&mut root.glyphs);
                root.glyphs = glyphs;
                root.procedures.extend(index.procedures);
                Ok(root)
            }
            Ast::Phantom(_) => Ok(Boxed {
                w: 0.7,
                above: 0.8,
                ..Boxed::empty()
            }),
            Ast::Enclosure(role, inner) => {
                let mut b = self.layout(inner)?;
                b.move_by(0.2, 0.);
                b.w += 0.4;
                b.above += 0.2;
                b.below += 0.2;
                let points = if role == "circle" {
                    (0..=48)
                        .map(|i| {
                            let t = i as f32 / 48. * std::f32::consts::TAU;
                            [
                                b.w * 0.5 + b.w * 0.55 * t.cos(),
                                (b.below - b.above) * 0.5 + (b.above + b.below) * 0.55 * t.sin(),
                            ]
                        })
                        .collect()
                } else {
                    vec![
                        [0., -b.above],
                        [b.w, -b.above],
                        [b.w, b.below],
                        [0., b.below],
                        [0., -b.above],
                    ]
                };
                b.procedures.push(Procedure {
                    role: role.clone(),
                    mode: "procedural".into(),
                    points,
                });
                Ok(b)
            }
            Ast::Accent(role, inner) => {
                let mut b = self.layout(inner)?;
                b.above += 0.15;
                let y = -b.above;
                let points = if role == "vec" {
                    vec![
                        [0., y],
                        [b.w, y],
                        [b.w - 0.15, y - 0.08],
                        [b.w, y],
                        [b.w - 0.15, y + 0.08],
                    ]
                } else if role == "hat" {
                    vec![[0., y], [b.w * 0.5, y - 0.12], [b.w, y]]
                } else {
                    vec![[0., y], [b.w, y]]
                };
                b.procedures.push(Procedure {
                    role: role.clone(),
                    mode: "procedural".into(),
                    points,
                });
                Ok(b)
            }
            Ast::Table(kind, rows) => {
                let mut cells = rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|c| self.layout(c))
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                let cols = cells.iter().map(Vec::len).max().unwrap_or(0);
                ensure!(cols > 0, "empty table");
                if !["aligned", "gathered", "cases"].contains(&kind.as_str()) {
                    ensure!(cells.iter().all(|r| r.len() == cols), "ragged matrix");
                }
                let mut widths = vec![0.0_f32; cols];
                for row in &cells {
                    for (j, c) in row.iter().enumerate() {
                        widths[j] = widths[j].max(c.w);
                    }
                }
                let inset = 0.0;
                let w = widths.iter().sum::<f32>() + (cols - 1) as f32 * 0.5 + 2. * inset;
                let heights = cells
                    .iter()
                    .map(|r| {
                        let above = r.iter().map(|c| c.above).fold(0., f32::max);
                        let below = r.iter().map(|c| c.below).fold(0., f32::max);
                        (above, below)
                    })
                    .collect::<Vec<_>>();
                let total = heights.iter().map(|(a, b)| a + b + 0.4).sum::<f32>() - 0.4;
                let mut out = Boxed {
                    w,
                    above: total * 0.5 + 0.25,
                    below: (total * 0.5 - 0.25).max(0.),
                    ..Boxed::empty()
                };
                let mut y = -out.above;
                for (i, row) in cells.iter_mut().enumerate() {
                    y += heights[i].0;
                    let mut x = inset;
                    for (j, c) in row.drain(..).enumerate() {
                        let offset = if kind == "aligned" {
                            if j % 2 == 0 { widths[j] - c.w } else { 0. }
                        } else if kind == "cases" {
                            0.
                        } else {
                            (widths[j] - c.w) * 0.5
                        };
                        let mut c = c;
                        c.move_by(x + offset, y);
                        out.append(c);
                        x += widths[j] + 0.5;
                    }
                    y += heights[i].1 + 0.4;
                }
                let (left, right) = table_delimiters(kind);
                if left == "." && right == "." {
                    Ok(out)
                } else {
                    self.enclose(out, left, right)
                }
            }
            Ast::Scripts { base, sup, sub } => {
                let mut b = self.layout(base)?;
                let start = b.w;
                let mut extra: f32 = 0.;
                if let Some(s) = sup {
                    let mut s = self.layout(s)?;
                    s.scale(0.6);
                    let y = -b.above * 0.75 - s.below;
                    s.move_by(start, y);
                    b.above = b.above.max(s.above - y);
                    extra = extra.max(s.w);
                    b.append(s);
                }
                if let Some(s) = sub {
                    let mut s = self.layout(s)?;
                    s.scale(0.6);
                    let y = b.below + 0.3 + s.above * 0.5;
                    s.move_by(start, y);
                    b.below = b.below.max(y + s.below);
                    extra = extra.max(s.w);
                    b.append(s);
                }
                b.w += extra;
                Ok(b)
            }
        }
    }
}
pub fn generate(
    input: &str,
    seed: u64,
    provider: &mut dyn FnMut(&str, u64) -> Result<Glyph>,
) -> Result<Drawing> {
    generate_with_provider(input, seed, &mut |label: &str, local_seed| {
        provider(label, local_seed)
    })
}
pub fn generate_with_provider(
    input: &str,
    seed: u64,
    provider: &mut dyn InkProvider,
) -> Result<Drawing> {
    let ast = parse(input)?;
    let mut b = Layout {
        provider,
        rng: seed,
    }
    .layout(&ast)?;
    finish_drawing(input, seed, &mut b)
}

/// Complete sequence geometry is placed once. The decoder owns every mark,
/// including fractions, radicals, accents and enclosures; no AST path is drawn.
pub fn generate_sequence(input: &str, seed: u64, glyph: Glyph) -> Result<Drawing> {
    let mut expected = vec![];
    labels(&parse(input)?, &mut expected);
    ensure!(
        glyph.run_labels == expected,
        "sequence decoder changed requested content"
    );
    let mut b = Layout::run_box(glyph)?;
    finish_drawing(input, seed, &mut b)
}

fn finish_drawing(input: &str, seed: u64, b: &mut Boxed) -> Result<Drawing> {
    ensure!(
        !b.glyphs.is_empty() || !b.procedures.is_empty(),
        "empty drawing"
    );
    let mut bounds = [0.0_f32, -b.above, b.w, b.below];
    for g in &b.glyphs {
        for s in &g.strokes {
            for p in &s.points {
                let x = p[0] * g.transform[0] + g.transform[2];
                let y = p[1] * g.transform[1] + g.transform[3];
                let pad = PEN_WIDTH * 0.5;
                bounds[0] = bounds[0].min(x - pad);
                bounds[1] = bounds[1].min(y - pad);
                bounds[2] = bounds[2].max(x + pad);
                bounds[3] = bounds[3].max(y + pad);
            }
        }
    }
    for p in b.procedures.iter().flat_map(|p| &p.points) {
        bounds[0] = bounds[0].min(p[0] - PEN_WIDTH * 0.5);
        bounds[1] = bounds[1].min(p[1] - PEN_WIDTH * 0.5);
        bounds[2] = bounds[2].max(p[0] + PEN_WIDTH * 0.5);
        bounds[3] = bounds[3].max(p[1] + PEN_WIDTH * 0.5);
    }
    b.move_by(0.25 - bounds[0], 0.25 - bounds[1]);
    b.w = bounds[2] - bounds[0];
    b.above = bounds[3] - bounds[1];
    b.below = 0.;
    let drawing = Drawing {
        schema_version: 1,
        input: input.into(),
        seed,
        width: b.w + 0.5,
        height: b.above + b.below + 0.5,
        pen_width: Some(PEN_WIDTH),
        glyphs: std::mem::take(&mut b.glyphs),
        procedures: std::mem::take(&mut b.procedures),
    };
    ensure!(
        drawing.width.is_finite() && drawing.height.is_finite() && drawing.width < 100000.,
        "layout bounds invalid"
    );
    Ok(drawing)
}
pub fn svg(d: &Drawing) -> Result<String> {
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">\n<g fill=\"none\" stroke=\"#17202a\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n",
        d.width,
        d.height,
        d.width * 48.,
        d.height * 48.
    );
    for g in &d.glyphs {
        let [sx, sy, x, y] = g.transform;
        for s in &g.strokes {
            let p = |p: [f32; 2]| [p[0] * sx + x, p[1] * sy + y];
            let first = p(s.points[0]);
            let mut path = format!("M {} {}", first[0], first[1]);
            if s.points.len() == 1 {
                path.push_str(&format!(" L {} {}", first[0], first[1]));
            }
            for c in s.points[1..].as_chunks::<3>().0.iter() {
                let a = p(c[0]);
                let b = p(c[1]);
                let c = p(c[2]);
                path.push_str(&format!(
                    " C {} {} {} {} {} {}",
                    a[0], a[1], b[0], b[1], c[0], c[1]
                ));
            }
            out.push_str(&format!(
                "<path d=\"{path}\" stroke-width=\"{}\"/>\n",
                d.pen_width.unwrap_or((s.width * sx).clamp(0.015, 0.06))
            ));
        }
    }
    for p in &d.procedures {
        out.push_str(&format!(
            "<polyline stroke-width=\"{}\" points=\"",
            d.pen_width.unwrap_or(PEN_WIDTH)
        ));
        for q in &p.points {
            out.push_str(&format!("{},{} ", q[0], q[1]));
        }
        out.push_str("\"/>\n");
    }
    out.push_str("</g></svg>\n");
    Ok(out)
}
pub fn labels(ast: &Ast, out: &mut Vec<String>) {
    match ast {
        Ast::Glyph(s) => out.push(s.clone()),
        Ast::Space => {}
        Ast::Row(v) => {
            for x in v {
                labels(x, out)
            }
        }
        Ast::Fraction(a, b) => {
            labels(a, out);
            labels(b, out)
        }
        Ast::Root(a) | Ast::Enclosure(_, a) | Ast::Accent(_, a) => labels(a, out),
        Ast::Delimited(left, right, a) => {
            out.extend(delimiter_labels(left));
            labels(a, out);
            out.extend(delimiter_labels(right));
        }
        Ast::Phantom(_) => {}
        Ast::IndexedRoot(a, b) => {
            labels(a, out);
            labels(b, out);
        }
        Ast::Table(kind, rows) => {
            let (left, right) = table_delimiters(kind);
            out.extend(delimiter_labels(left));
            for row in rows {
                for cell in row {
                    labels(cell, out);
                }
            }
            out.extend(delimiter_labels(right));
        }
        Ast::Scripts { base, sup, sub } => {
            labels(base, out);
            if let Some(a) = sup {
                labels(a, out)
            }
            if let Some(a) = sub {
                labels(a, out)
            }
        }
    }
}
/// Semantic conditions, shared by training and inference; these are not stroke labels.
pub fn tokens(ast: &Ast, out: &mut Vec<String>) {
    match ast {
        Ast::Glyph(s) => out.push(s.clone()),
        Ast::Space => out.push("<space>".into()),
        Ast::Row(v) => {
            for a in v {
                tokens(a, out)
            }
        }
        Ast::Fraction(a, b) => {
            out.push("<frac>".into());
            tokens(a, out);
            out.push("<denominator>".into());
            tokens(b, out);
            out.push("</frac>".into());
        }
        Ast::Root(a) => {
            out.push("<sqrt>".into());
            tokens(a, out);
            out.push("</sqrt>".into());
        }
        Ast::IndexedRoot(i, a) => {
            out.push("<root-index>".into());
            tokens(i, out);
            out.push("<sqrt>".into());
            tokens(a, out);
            out.push("</sqrt>".into());
        }
        Ast::Delimited(left, right, a) => {
            out.extend(delimiter_labels(left));
            tokens(a, out);
            out.extend(delimiter_labels(right));
        }
        Ast::Phantom(_) => out.push("<phantom>".into()),
        Ast::Enclosure(role, a) | Ast::Accent(role, a) => {
            out.push(format!("<{role}>"));
            tokens(a, out);
            out.push(format!("</{role}>"));
        }
        Ast::Scripts { base, sup, sub } => {
            tokens(base, out);
            if let Some(a) = sup {
                out.push("<sup>".into());
                tokens(a, out);
                out.push("</sup>".into());
            }
            if let Some(a) = sub {
                out.push("<sub>".into());
                tokens(a, out);
                out.push("</sub>".into());
            }
        }
        Ast::Table(kind, rows) => {
            let (left, right) = table_delimiters(kind);
            out.extend(delimiter_labels(left));
            out.push(format!("<{kind}>"));
            for row in rows {
                out.push("<row>".into());
                for a in row {
                    out.push("<cell>".into());
                    tokens(a, out);
                }
            }
            out.push(format!("</{kind}>"));
            out.extend(delimiter_labels(right));
        }
    }
}
pub fn tokenize(input: &str) -> Result<Vec<String>> {
    let mut out = vec![];
    tokens(&parse(input)?, &mut out);
    ensure!(!out.is_empty(), "empty model condition");
    Ok(out)
}

/// Shared em-space metrics for training, neural ink conversion and layout.
pub fn font_metrics(label: &str) -> (f32, f32) {
    match label {
        "," => (0.12, 0.12),
        "." => (0.08, 0.),
        "\\cdot" => (0.42, -0.38),
        ":" => (0.4, 0.),
        ";" => (0.3, 0.1),
        "'" | "\"" => (0.95, -0.55),
        "-" => (0.45, -0.35),
        "=" | "\\circ" => (0.525, -0.275),
        "+" => (0.8, 0.),
        "*" | "\\times" | "\\div" | "\\pm" | "\\leq" | "\\geq" | "\\neq" | "\\approx" | "\\sim"
        | "\\propto" | "\\in" => (0.7, -0.1),
        _ => {
            if label.chars().count() == 1 {
                let c = label.chars().next().unwrap();
                if "gjpqy".contains(c) {
                    (0.72, 0.23)
                } else if "áéíóúñ".contains(c) {
                    (0.95, 0.)
                } else if c.is_lowercase() && !"bdfhklt".contains(c) {
                    (0.72, 0.)
                } else {
                    (1., 0.)
                }
            } else if [
                "\\alpha",
                "\\beta",
                "\\gamma",
                "\\delta",
                "\\epsilon",
                "\\theta",
                "\\lambda",
                "\\mu",
                "\\pi",
                "\\rho",
                "\\sigma",
                "\\tau",
                "\\phi",
                "\\omega",
                "\\eta",
                "\\upsilon",
            ]
            .contains(&label)
            {
                (0.8, 0.)
            } else {
                (1., 0.)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whitelist() {
        for s in [
            "x^2",
            "x_{i}^{2}",
            "\\frac{x+1}{y}",
            "\\sqrt{x}",
            "\\text{abc}",
        ] {
            assert!(parse(s).is_ok(), "{s}");
        }
        for s in [
            "\\input{file}",
            "x^^2",
            "{x",
            "x}",
            "\\unknown",
            "\\text{a\\b}",
            "^x",
        ] {
            assert!(parse(s).is_err(), "{s}");
        }
    }
}

#[cfg(test)]
mod contracts {
    use super::*;
    #[test]
    fn one_pen_survives_nested_layout_and_learned_symbols() {
        let mut provider = |label: &str, seed| -> Result<Glyph> {
            ensure!(!procedural_symbol(label), "dots must bypass the decoder");
            ensure!(
                neural_condition(label) == label,
                "composed labels must use donors"
            );
            Ok(Glyph {
                run_labels: vec![],
                condition_label: None,
                label: label.into(),
                mode: "neural".into(),
                model_id: "fixture".into(),
                decoder_sha256: "a".repeat(64),
                latent_seed: seed,
                transform: [1., 1., 0., 0.],
                strokes: vec![Stroke {
                    points: vec![[0., 0.], [0., 1.], [1., 1.], [1., 0.]],
                    source_index: 0,
                    uuid: "generated".into(),
                    width: if label == "x" { 0.06 } else { 0.015 },
                }],
            })
        };
        let drawing = generate(
            r"x\cdot\frac{x^2\cdot y;}{\frac{y}{x}}\text{áéóúÉñ¿}\downarrow\in\propto",
            419,
            &mut provider,
        )
        .unwrap();
        let rendered = svg(&drawing).unwrap();
        for width in rendered.split("stroke-width=\"").skip(1) {
            assert_eq!(
                width.split('"').next().unwrap(),
                "0.025",
                "same pen at every depth"
            );
        }
        let dot = drawing.glyphs.iter().find(|g| g.label == r"\cdot").unwrap();
        assert_eq!(dot.mode, "neural");
        assert_eq!(dot.strokes[0].uuid, "generated");
        for glyph in &drawing.glyphs {
            assert_eq!(glyph.mode, "neural");
            for stroke in &glyph.strokes {
                assert_eq!((stroke.points.len() - 1) % 3, 0, "complete cubic pen paths");
            }
        }
    }
    #[test]
    fn layout_preserves_order_geometry_and_provider_failures() {
        let mut provider = |label: &str, seed| -> Result<Glyph> {
            ensure!(label != "z", "missing glyph z");
            Ok(Glyph {
                run_labels: vec![],
                condition_label: None,
                label: label.into(),
                mode: "neural".into(),
                model_id: "fixture".into(),
                decoder_sha256: "a".repeat(64),
                latent_seed: seed,
                transform: [1., 1., 0., 0.],
                strokes: vec![Stroke {
                    points: vec![[0., 0.], [0., 1.], [1., 1.], [1., 0.]],
                    source_index: 0,
                    uuid: "generated".into(),
                    width: 0.025,
                }],
            })
        };
        for input in ["x^2", "\\frac{x_1}{\\sqrt{y^2}}", "{xy}^{3}"] {
            let a = generate(input, 419, &mut provider).unwrap();
            let b = generate(input, 419, &mut provider).unwrap();
            assert_eq!(
                serde_json::to_vec(&a).unwrap(),
                serde_json::to_vec(&b).unwrap()
            );
            let mut expected = vec![];
            labels(&parse(input).unwrap(), &mut expected);
            assert_eq!(
                a.glyphs.iter().map(|g| g.label.clone()).collect::<Vec<_>>(),
                expected
            );
            for g in a.glyphs {
                for p in g.strokes.iter().flat_map(|s| &s.points) {
                    let x = p[0] * g.transform[0] + g.transform[2];
                    let y = p[1] * g.transform[1] + g.transform[3];
                    assert!(x >= 0. && x <= a.width && y >= 0. && y <= a.height);
                }
            }
        }
        assert!(generate("z", 0, &mut provider).is_err());
    }
}

#[cfg(test)]
mod extended_contracts {
    use super::*;
    #[test]
    fn structured_dataset_grammar() {
        for input in [
            r"\boxed{Hello, World!}",
            r"\begin{bmatrix}1&2\\3&4\end{bmatrix}",
            r"\begin{cases}x=1\\y=2\end{cases}",
            r"\sqrt[3]{\phantom{x}}",
            r"\frac13 36\sqrt2",
            r"\begin{aligned}x&=2\\y&=3\end{aligned}",
            r"\text{¿Está? Éáéíñóú}",
            r"\enclose{circle}{x}",
            r"\vec{x}_1",
            r"\begin{array}{cc}1&2\\3&4\end{array}",
            r"\left(\frac{x}{y}\right)",
        ] {
            assert!(parse(input).is_ok(), "{input}");
        }
        for input in [
            r"\begin{bmatrix}1&2\end{cases}",
            r"\begin{bmatrix}1&2",
            r"\sqrt[3{x}",
            r"\write18{a}",
        ] {
            assert!(parse(input).is_err(), "{input}");
        }
    }
}

#[cfg(test)]
mod handwriting_revision_contracts {
    use super::*;
    fn glyph(label: &str, seed: u64) -> Result<Glyph> {
        Ok(Glyph {
            run_labels: vec![],
            condition_label: Some(label.into()),
            label: label.into(),
            mode: "neural".into(),
            model_id: "fixture".into(),
            decoder_sha256: "a".repeat(64),
            latent_seed: seed,
            strokes: vec![Stroke {
                points: vec![[0., 0.], [0.2, 0.1], [0.1, 0.8], [0.3, 1.]],
                source_index: 0,
                uuid: "learned-fixture".into(),
                width: PEN_WIDTH,
            }],
            transform: [1., 1., 0., 0.],
        })
    }
    #[test]
    fn delimiters_and_punctuation_reach_the_neural_provider() {
        for (input, expected) in [
            (r"\left( x \right)", vec!["(", "x", ")"]),
            (
                r"\begin{bmatrix}x&y\\1&2\end{bmatrix}",
                vec!["[", "x", "y", "1", "2", "]"],
            ),
            (r"\begin{cases}x\\y\end{cases}", vec!["{", "x", "y"]),
            (r"\cdot;\in\propto", vec![r"\cdot", ";", r"\in", r"\propto"]),
        ] {
            let mut requested = vec![];
            let d = generate(input, 419, &mut |label, seed| {
                requested.push(label.to_owned());
                glyph(label, seed)
            })
            .unwrap();
            let mut got = requested;
            got.sort();
            let mut want = expected.clone();
            want.sort();
            assert_eq!(
                got, want,
                "every requested character must use its own learned condition"
            );
            assert_eq!(drawing_labels(&d), expected);
            assert!(
                d.procedures.is_empty(),
                "no procedural delimiter or punctuation"
            );
            assert!(
                d.glyphs
                    .iter()
                    .all(|g| g.strokes.iter().all(|s| s.uuid == "learned-fixture"))
            );
        }
    }
    #[test]
    fn a_complete_run_preserves_joined_curves_drift_and_delayed_strokes() {
        let mut g = glyph("Tt", 419).unwrap();
        g.run_labels = vec!["T".into(), "t".into()];
        g.mode = "neural-run".into();
        // A cross-letter trajectory and a later crossbar; no artificial split or join.
        g.strokes[0].points = vec![
            [0., 0.],
            [0.3, 0.9],
            [0.8, 1.2],
            [1.4, 0.3],
            [1.6, 0.4],
            [1.8, 1.4],
            [2., 1.1],
        ];
        let mut crossbar = g.strokes[0].clone();
        crossbar.points = vec![[0.1, 0.2], [0.4, 0.25], [0.9, 0.15], [1.2, 0.3]];
        crossbar.source_index = 1;
        g.strokes.push(crossbar);
        let original = serde_json::to_vec(&g.strokes).unwrap();
        let d = generate_sequence("Tt", 419, g).unwrap();
        assert!(d.procedures.is_empty());
        assert_eq!(d.glyphs.len(), 1);
        assert_eq!(serde_json::to_vec(&d.glyphs[0].strokes).unwrap(), original);
        assert_eq!(d.glyphs[0].transform[0], d.glyphs[0].transform[1]);
        assert_eq!(drawing_labels(&d), ["T", "t"]);
    }
    #[test]
    fn complete_math_sequence_creates_no_procedural_paths() {
        let input = r"\boxed{\left[\frac{T}{\sqrt{t}}\right]}";
        let mut g = glyph(input, 419).unwrap();
        labels(&parse(input).unwrap(), &mut g.run_labels);
        g.mode = "neural-run".into();
        let d = generate_sequence(input, 419, g).unwrap();
        assert!(d.procedures.is_empty());
        assert_eq!(drawing_labels(&d), ["[", "T", "t", "]"]);
        let mut expected = vec![];
        labels(
            &parse(r"\begin{Vmatrix}T\end{Vmatrix}").unwrap(),
            &mut expected,
        );
        assert_eq!(expected, ["|", "|", "T", "|", "|"]);
    }
    #[test]
    fn the_same_letter_pair_can_continue_or_lift_without_forced_connections() {
        let first = vec![[0., 0.], [0.2, 0.9], [0.6, 1.], [1., 0.7]];
        let second = vec![[1., 0.7], [1.3, 0.3], [1.8, 0.9], [2., 1.1]];
        for joined in [false, true] {
            let mut g = glyph("th", 419).unwrap();
            g.run_labels = vec!["t".into(), "h".into()];
            g.mode = "neural-run".into();
            g.strokes[0].points = first.clone();
            if joined {
                g.strokes[0].points.extend_from_slice(&second[1..]);
            } else {
                let mut lifted = g.strokes[0].clone();
                lifted.points = second.clone();
                lifted.source_index = 1;
                g.strokes.push(lifted);
            }
            let sample = ink_corpus::Sample {
                sample_id: "synthetic-contract-only".into(),
                label: "th".into(),
                source_sha256: "a".repeat(64),
                split: "train".into(),
                metadata: serde_json::Value::Null,
                strokes: g.strokes,
            };
            let encoded = ink_corpus::sequence::encode(&sample).unwrap();
            let moves = encoded
                .events
                .iter()
                .filter(|e| e[ink_corpus::sequence::MOVE] > 0.5)
                .count();
            assert_eq!(moves, if joined { 1 } else { 2 });
            g.strokes =
                ink_corpus::sequence::decode(&encoded.events, encoded.origin, encoded.scale)
                    .unwrap();
            let drawing = generate_sequence("th", 419, g).unwrap();
            assert!(drawing.procedures.is_empty());
            assert_eq!(drawing_labels(&drawing), ["t", "h"]);
            assert_eq!(drawing.glyphs[0].strokes.len(), moves);
            assert_eq!(svg(&drawing).unwrap().matches("<path ").count(), moves);
            for (actual, original) in drawing.glyphs[0].strokes.iter().zip(&sample.strokes) {
                assert_eq!(actual.points.len(), original.points.len());
                for (p, q) in actual.points.iter().zip(&original.points) {
                    assert!((p[0] - q[0]).abs() < 1e-5 && (p[1] - q[1]).abs() < 1e-5);
                }
            }
        }
    }
}
