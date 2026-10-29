//! Execution overrides are recorded separately from the architecture recipe.
//! Resume retains its original grouping so the saved Adam cursor remains exact.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct ExecutionPlan {
    pub max_batch_samples: usize,
    pub max_padded_events: Option<usize>,
}
impl ExecutionPlan {
    pub fn new(cpu: bool) -> Self {
        Self {
            max_batch_samples: if cpu { 8 } else { 64 },
            max_padded_events: Some(16_384),
        }
    }
    pub fn resume(manifest: &Value) -> Result<Self> {
        let plan = if let Some(value) = manifest.get("execution_plan") {
            serde_json::from_value(value.clone())?
        } else {
            // Historical v4 checkpoints used exactly eight sorted rows per batch.
            Self {
                max_batch_samples: 8,
                max_padded_events: None,
            }
        };
        ensure!(
            (1..=64).contains(&plan.max_batch_samples),
            "Invalid saved batch size"
        );
        ensure!(
            plan.max_padded_events.is_none_or(|n| n > 0 && n <= 16_384),
            "Invalid saved event budget"
        );
        Ok(plan)
    }
    pub fn groups(&self, lengths: &[usize]) -> Vec<Vec<usize>> {
        let mut order: Vec<_> = (0..lengths.len()).collect();
        order.sort_by_key(|&i| lengths[i]);
        let mut groups = vec![];
        let mut group = vec![];
        for i in order {
            // The next row is longest because order is sorted. An oversized
            // single row is retained whole rather than truncated or discarded.
            let padded = lengths[i].saturating_mul(group.len() + 1);
            if !group.is_empty()
                && (group.len() == self.max_batch_samples
                    || self.max_padded_events.is_some_and(|limit| padded > limit))
            {
                groups.push(std::mem::take(&mut group));
            }
            group.push(i);
        }
        if !group.is_empty() {
            groups.push(group);
        }
        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn throughput_batches_cover_every_row_and_bound_padding() {
        let lengths: Vec<_> = (1..=150).map(|i| i * 20).chain([20_000]).collect();
        let plan = ExecutionPlan::new(false);
        let groups = plan.groups(&lengths);
        let mut visited: Vec<_> = groups.iter().flatten().copied().collect();
        visited.sort();
        assert_eq!(visited, (0..lengths.len()).collect::<Vec<_>>());
        assert_eq!(groups, plan.groups(&lengths));
        for g in groups {
            assert!(g.len() <= 64);
            assert!(g.len() == 1 || g.len() * lengths[*g.last().unwrap()] <= 16_384);
        }
    }
    #[test]
    fn resume_preserves_historical_and_recorded_batch_boundaries() {
        let lengths: Vec<_> = (0..131).map(|i| 131 - i).collect();
        let old = ExecutionPlan::resume(&serde_json::json!({})).unwrap();
        assert_eq!(
            old.groups(&lengths)
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            (0..16).map(|_| 8).chain([3]).collect::<Vec<_>>()
        );
        let new = ExecutionPlan::new(false);
        let saved = serde_json::json!({"execution_plan":new});
        assert_eq!(
            ExecutionPlan::resume(&saved).unwrap().groups(&lengths),
            new.groups(&lengths)
        );
        assert!(
            ExecutionPlan::resume(
                &serde_json::json!({"execution_plan":{"max_batch_samples":0,"max_padded_events":1}})
            )
            .is_err()
        );
    }
}
