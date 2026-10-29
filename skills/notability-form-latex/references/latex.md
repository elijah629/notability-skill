# Supported math input

Pass raw LaTeX without dollar delimiters or a document preamble. This is a bounded handwritten layout parser, not a TeX engine. JSON strings require doubled backslashes; shell single-quoted strings preserve backslashes.

Useful examples:

```text
\frac{x^2+1}{\sqrt{y}}
\sqrt[3]{x}+\alpha+\pi
\left[\frac{x+1}{2}\right]
\boxed{Hello, World!}
\begin{bmatrix}1&2\\3&4\end{bmatrix}
x\cdot y=2
```

The parser includes fractions, indexed roots, superscripts/subscripts, stretchy delimiters, boxes, text, accents, matrices, cases, aligned, gathered and array structures. Individual symbol availability depends on the model's vocabulary. Unsupported commands and missing glyphs fail explicitly; use the exact failed symbol to revise the expression with the user, rather than silently changing its meaning. Test the actual expression before allocating its PDF box. Long prose needs multiple placements and visual inspection.

Fraction bars, radical outlines, matrix brackets and dots must remain present when transferring SVG to PDF. The bundled helper handles cubic paths, polylines and single-point marks. Output uses vector paths; mathematical structure is visually rendered rather than exposed as accessible semantic math.
