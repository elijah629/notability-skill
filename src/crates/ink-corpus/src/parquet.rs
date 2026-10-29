//! Deliberately narrow Parquet V1 reader for the supplied plain, required-scalar
//! tables. Reject compressed, nullable, dictionary and multi-page encodings.
use anyhow::{Result, bail, ensure};
use serde_json::Value;
use std::collections::BTreeMap;
#[derive(Debug)]
enum T {
    Int(i64),
    Bytes(Vec<u8>),
    List(Vec<T>),
    Struct(BTreeMap<i16, T>),
    Other,
}
impl T {
    fn int(&self) -> Result<i64> {
        if let Self::Int(v) = self {
            Ok(*v)
        } else {
            bail!("expected Thrift integer")
        }
    }
    fn list(&self) -> Result<&[T]> {
        if let Self::List(v) = self {
            Ok(v)
        } else {
            bail!("expected Thrift list")
        }
    }
    fn field(&self, k: i16) -> Result<&T> {
        if let Self::Struct(v) = self {
            v.get(&k)
                .ok_or_else(|| anyhow::anyhow!("missing Thrift field {k}"))
        } else {
            bail!("expected Thrift struct")
        }
    }
    fn string(&self) -> Result<String> {
        if let Self::Bytes(v) = self {
            Ok(String::from_utf8(v.clone())?)
        } else {
            bail!("expected Thrift string")
        }
    }
}
struct Compact<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Compact<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .ok_or_else(|| anyhow::anyhow!("Thrift overflow"))?;
        ensure!(end <= self.b.len(), "truncated Thrift");
        let v = &self.b[self.p..end];
        self.p = end;
        Ok(v)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn var(&mut self) -> Result<u64> {
        let mut n = 0;
        for shift in (0..64).step_by(7) {
            let x = self.byte()?;
            n |= ((x & 127) as u64) << shift;
            if x & 128 == 0 {
                return Ok(n);
            }
        }
        bail!("oversized Thrift integer")
    }
    fn value(&mut self, t: u8, depth: usize) -> Result<T> {
        ensure!(depth < 32, "Thrift nesting limit");
        Ok(match t {
            1 | 2 => T::Other,
            3 => T::Int(self.byte()? as i8 as i64),
            4..=6 => {
                let n = self.var()?;
                T::Int(((n >> 1) as i64) ^ -((n & 1) as i64))
            }
            7 => {
                self.take(8)?;
                T::Other
            }
            8 => {
                let n = usize::try_from(self.var()?)?;
                T::Bytes(self.take(n)?.to_vec())
            }
            9 | 10 => {
                let h = self.byte()?;
                let n = if h >> 4 == 15 {
                    usize::try_from(self.var()?)?
                } else {
                    (h >> 4) as usize
                };
                ensure!(n <= 1_000_000, "Thrift list limit");
                let mut v = Vec::new();
                for _ in 0..n {
                    v.push(self.value(h & 15, depth + 1)?)
                }
                T::List(v)
            }
            12 => {
                let mut v = BTreeMap::new();
                let mut previous = 0i16;
                loop {
                    let h = self.byte()?;
                    if h == 0 {
                        break;
                    }
                    let k = if h >> 4 == 0 {
                        self.value(4, depth + 1)?.int()? as i16
                    } else {
                        previous
                            .checked_add((h >> 4) as i16)
                            .ok_or_else(|| anyhow::anyhow!("field overflow"))?
                    };
                    ensure!(
                        v.insert(k, self.value(h & 15, depth + 1)?).is_none(),
                        "duplicate Thrift field"
                    );
                    previous = k
                }
                T::Struct(v)
            }
            _ => bail!("unsupported Thrift compact type {t}"),
        })
    }
}
pub fn rows(b: &[u8]) -> Result<Vec<BTreeMap<String, Value>>> {
    ensure!(
        b.len() >= 12 && &b[..4] == b"PAR1" && &b[b.len() - 4..] == b"PAR1",
        "invalid Parquet magic"
    );
    let len = u32::from_le_bytes(b[b.len() - 8..b.len() - 4].try_into()?) as usize;
    ensure!(len <= b.len() - 12, "invalid Parquet footer");
    let footer = b.len() - 8 - len;
    let mut c = Compact {
        b: &b[footer..b.len() - 8],
        p: 0,
    };
    let meta = c.value(12, 0)?;
    let n = usize::try_from(meta.field(3)?.int()?)?;
    ensure!(n <= 1_000_000, "row limit");
    let schema = meta.field(2)?.list()?;
    let groups = meta.field(4)?.list()?;
    ensure!(
        groups.len() == 1,
        "only single row-group dataset tables supported"
    );
    let columns = groups[0].field(1)?.list()?;
    ensure!(columns.len() + 1 == schema.len(), "Parquet schema mismatch");
    let mut output = vec![BTreeMap::new(); n];
    for (column, s) in columns.iter().zip(&schema[1..]) {
        ensure!(s.field(3)?.int()? == 0, "nullable Parquet not supported");
        let name = s.field(4)?.string()?;
        let typ = s.field(1)?.int()?;
        let m = column.field(3)?;
        ensure!(
            m.field(4)?.int()? == 0 && m.field(5)?.int()? == n as i64,
            "compressed/inconsistent Parquet column"
        );
        let offset = usize::try_from(m.field(9)?.int()?)?;
        ensure!(offset < footer, "bad page offset");
        let mut page = Compact {
            b: &b[offset..footer],
            p: 0,
        };
        let header = page.value(12, 0)?;
        ensure!(
            header.field(1)?.int()? == 0 && header.field(2)?.int()? == header.field(3)?.int()?,
            "unsupported Parquet page"
        );
        let data = header.field(5)?;
        ensure!(
            data.field(1)?.int()? == n as i64 && data.field(2)?.int()? == 0,
            "only single PLAIN page supported"
        );
        let size = usize::try_from(header.field(3)?.int()?)?;
        let bytes = page.take(size)?;
        let mut plain = Compact { b: bytes, p: 0 };
        for row in &mut output {
            let v = match typ {
                6 => {
                    let k = u32::from_le_bytes(plain.take(4)?.try_into()?) as usize;
                    Value::String(String::from_utf8(plain.take(k)?.to_vec())?)
                }
                2 => Value::from(i64::from_le_bytes(plain.take(8)?.try_into()?)),
                1 => Value::from(i32::from_le_bytes(plain.take(4)?.try_into()?)),
                5 => {
                    let f = f64::from_le_bytes(plain.take(8)?.try_into()?);
                    ensure!(f.is_finite(), "nonfinite Parquet scalar");
                    Value::from(f)
                }
                _ => bail!("unsupported Parquet physical type {typ}"),
            };
            row.insert(name.clone(), v);
        }
        ensure!(plain.p == bytes.len(), "extra Parquet page data");
    }
    Ok(output)
}
