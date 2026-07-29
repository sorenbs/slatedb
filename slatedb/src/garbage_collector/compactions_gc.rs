//! Garbage collection for `.compactions` files.
//!
//! The compactions store is a versioned log of compactor state snapshots. GC can
//! safely delete older versions because only the latest record is required for:
//! - establishing the compaction low-watermark (to prevent deleting in-flight outputs),
//! - fencing semantics (epoch checks on the newest record).
//!
//! Policy:
//! - Always retain the most recent `.compactions` file.
//! - Only delete files older than the configured `min_age`.
//!
//! Safety:
//! - Deleting old versions does not affect recovery because the newest record
//!   contains the authoritative compactor epoch and retained compaction state.
//! - This task does not inspect compaction contents; it is purely time-based and
//!   version-aware (keeps the latest).
//!
//! Errors are logged and the task continues; stats are updated only on successful
//! deletes.

use crate::{
    compactions_store::CompactionsStore, config::GarbageCollectorDirectoryOptions,
    error::SlateDBError,
};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use log::error;
use std::sync::Arc;

use super::filter::retain_allowed_by_gc_filter;
use super::{CachedDirListing, GcFilter, GcStats, GcTask, GC_DELETE_CONCURRENCY};
use slatedb_common::object_metadata::IdentifiedObjectMetadata;
use std::collections::HashSet;

#[derive(Clone)]
pub(crate) struct CompactionsGcTask {
    compactions_store: Arc<CompactionsStore>,
    stats: Arc<GcStats>,
    compactions_options: GarbageCollectorDirectoryOptions,
    gc_filter: Option<Arc<dyn GcFilter>>,
    boundary_files_enabled: bool,
    dir_listing: CachedDirListing<IdentifiedObjectMetadata<u64>>,
}

impl std::fmt::Debug for CompactionsGcTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactionsGcTask")
            .field("compactions_options", &self.compactions_options)
            .field("boundary_files_enabled", &self.boundary_files_enabled)
            .finish()
    }
}

impl CompactionsGcTask {
    pub(super) fn new(
        compactions_store: Arc<CompactionsStore>,
        stats: Arc<GcStats>,
        compactions_options: GarbageCollectorDirectoryOptions,
        gc_filter: Option<Arc<dyn GcFilter>>,
        boundary_files_enabled: bool,
    ) -> Self {
        Self {
            compactions_store,
            stats,
            compactions_options,
            gc_filter,
            boundary_files_enabled,
            dir_listing: CachedDirListing::new(),
        }
    }

    fn compactions_min_age(&self) -> chrono::Duration {
        chrono::Duration::from_std(self.compactions_options.min_age).expect("invalid duration")
    }

    /// Deletes the given compactions files from the compactions store.
    ///
    /// In case of dryrun, the actual deletion doesn't happen.
    async fn maybe_delete_compactions(&self, compactions_ids: Vec<u64>) {
        if self.compactions_options.dry_run {
            if !compactions_ids.is_empty() {
                log::info!(
                    "dry run: skipping compactions deletion [count={}]",
                    compactions_ids.len()
                );
            }
            for id in compactions_ids {
                log::debug!(
                    "dry run: would delete compactions but skipped [id={:?}]",
                    id
                );
            }
            return;
        }

        futures::stream::iter(compactions_ids)
            .for_each_concurrent(GC_DELETE_CONCURRENCY, |id| async move {
                if let Err(e) = self
                    .compactions_store
                    .delete_compactions_unchecked(id)
                    .await
                {
                    error!("error deleting compactions [id={:?}, error={}]", id, e);
                } else {
                    self.stats.gc_compactions_count.increment(1);
                }
            })
            .await;
    }
}

impl GcTask for CompactionsGcTask {
    /// Collect garbage from the compactions store. This will delete any compactions files
    /// that are older than the minimum age specified in the options, excluding the latest
    /// compactions file.
    async fn collect(&self, utc_now: DateTime<Utc>) -> Result<usize, SlateDBError> {
        let min_age = self.compactions_min_age();
        // The never-delete anchor (the latest compactions file) is read fresh
        // every sweep; only the candidate inventory may come from a cached
        // listing (list_cache_ttl). No latest record means nothing to judge.
        let Some(latest) = self.compactions_store.try_read_latest_compactions().await? else {
            return Ok(0);
        };
        let refresh_floor = (self
            .compactions_options
            .interval
            .unwrap_or(super::DEFAULT_INTERVAL)
            * 2)
        .max(std::time::Duration::from_secs(60));
        let compactions_metadata_list = self
            .dir_listing
            .entries(
                utc_now,
                self.compactions_options.list_cache_ttl,
                refresh_floor,
                self.compactions_store.list_compactions(..),
            )
            .await?;

        // Delete compactions files older than min_age, never the latest
        let compactions_to_delete = compactions_metadata_list
            .into_iter()
            .filter(|compactions_metadata| {
                compactions_metadata.id < latest.id
                    && utc_now.signed_duration_since(compactions_metadata.metadata.last_modified)
                        > min_age
            })
            .collect::<Vec<_>>();
        self.dir_listing
            .note_sweep_at(utc_now, !compactions_to_delete.is_empty());

        // Advance the boundary to the latest compactions file selected by the GC model. The
        // optional GC filter only gates the final deletion pass.
        if self.boundary_files_enabled {
            if let Some(boundary) = compactions_to_delete
                .iter()
                .map(|compactions_metadata| compactions_metadata.id)
                .max()
            {
                self.compactions_store.advance_boundary(boundary).await?;
            }
        }
        let compactions_to_delete =
            retain_allowed_by_gc_filter(&self.gc_filter, compactions_to_delete).await;
        let found = compactions_to_delete.len();
        let compactions_ids_to_delete = compactions_to_delete
            .into_iter()
            .map(|compactions_metadata| compactions_metadata.id)
            .collect::<Vec<_>>();

        self.maybe_delete_compactions(compactions_ids_to_delete.clone())
            .await;
        // Attempted deletions leave the cached view; a failed delete
        // resurfaces at the next listing refresh instead of retrying
        // from a stale entry.
        self.dir_listing.forget(|m| compactions_ids_to_delete.contains(&m.id));

        Ok(found)
    }

    fn resource(&self) -> &str {
        "Compactions"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compactions_store::{CompactionsStore, StoredCompactions};
    use async_trait::async_trait;
    use chrono::TimeDelta;
    use object_store::{memory::InMemory, path::Path, ObjectStoreExt};
    use slatedb_common::metrics::MetricsRecorderHelper;
    use slatedb_common::ObjectMetadata;
    use std::collections::HashSet;
    use std::time::Duration;

    struct DenyAllGcFilter;

    #[async_trait]
    impl GcFilter for DenyAllGcFilter {
        async fn filter(&self, _candidates: HashSet<ObjectMetadata>) -> HashSet<ObjectMetadata> {
            HashSet::new()
        }
    }

    #[tokio::test]
    async fn test_collect_advances_boundary_for_old_compactions_files() {
        let object_store = Arc::new(InMemory::new());
        let compactions_store = Arc::new(CompactionsStore::new(
            &Path::from("/root"),
            object_store.clone(),
        ));
        let mut stored_compactions = StoredCompactions::create(compactions_store.clone(), 0)
            .await
            .unwrap();
        stored_compactions
            .update(stored_compactions.prepare_dirty().unwrap())
            .await
            .unwrap();
        stored_compactions
            .update(stored_compactions.prepare_dirty().unwrap())
            .await
            .unwrap();

        let recorder = MetricsRecorderHelper::noop();
        let task = CompactionsGcTask::new(
            compactions_store.clone(),
            Arc::new(GcStats::new(&recorder)),
            GarbageCollectorDirectoryOptions {
                min_age: Duration::from_secs(1),
                interval: None,
                dry_run: false,
                max_interval: None,
                list_cache_ttl: None,
            },
            None,
            true,
        );
        task.collect(Utc::now() + TimeDelta::hours(1))
            .await
            .unwrap();

        let raw_boundary = object_store
            .get(&Path::from("/root/gc/compactions.boundary"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!("2", std::str::from_utf8(&raw_boundary).unwrap());

        let compactions = compactions_store.list_compactions(..).await.unwrap();
        assert_eq!(
            vec![3],
            compactions
                .iter()
                .map(|compactions| compactions.id)
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_collect_without_boundary_advancement_deletes_and_preserves_boundary() {
        let object_store = Arc::new(InMemory::new());
        let compactions_store = Arc::new(CompactionsStore::new(
            &Path::from("/root"),
            object_store.clone(),
        ));
        let mut stored_compactions = StoredCompactions::create(compactions_store.clone(), 0)
            .await
            .unwrap();
        stored_compactions
            .update(stored_compactions.prepare_dirty().unwrap())
            .await
            .unwrap();
        compactions_store.advance_boundary(1).await.unwrap();
        stored_compactions
            .update(stored_compactions.prepare_dirty().unwrap())
            .await
            .unwrap();

        let recorder = MetricsRecorderHelper::noop();
        let task = CompactionsGcTask::new(
            compactions_store.clone(),
            Arc::new(GcStats::new(&recorder)),
            GarbageCollectorDirectoryOptions {
                min_age: Duration::from_secs(1),
                interval: None,
                dry_run: false,
            },
            None,
            false,
        );
        task.collect(Utc::now() + TimeDelta::hours(1))
            .await
            .unwrap();

        let raw_boundary = object_store
            .get(&Path::from("/root/gc/compactions.boundary"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!("1", std::str::from_utf8(&raw_boundary).unwrap());
        let compactions = compactions_store.list_compactions(..).await.unwrap();
        assert_eq!(
            vec![3],
            compactions
                .iter()
                .map(|compactions| compactions.id)
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_collect_advances_boundary_before_filtering_compactions_files() {
        let object_store = Arc::new(InMemory::new());
        let compactions_store = Arc::new(CompactionsStore::new(
            &Path::from("/root"),
            object_store.clone(),
        ));
        let mut stored_compactions = StoredCompactions::create(compactions_store.clone(), 0)
            .await
            .unwrap();
        stored_compactions
            .update(stored_compactions.prepare_dirty().unwrap())
            .await
            .unwrap();
        stored_compactions
            .update(stored_compactions.prepare_dirty().unwrap())
            .await
            .unwrap();

        let recorder = MetricsRecorderHelper::noop();
        let task = CompactionsGcTask::new(
            compactions_store.clone(),
            Arc::new(GcStats::new(&recorder)),
            GarbageCollectorDirectoryOptions {
                min_age: Duration::from_secs(1),
                interval: None,
                dry_run: false,
                max_interval: None,
                list_cache_ttl: None,
            },
            Some(Arc::new(DenyAllGcFilter) as Arc<dyn GcFilter>),
            true,
        );
        task.collect(Utc::now() + TimeDelta::hours(1))
            .await
            .unwrap();

        let raw_boundary = object_store
            .get(&Path::from("/root/gc/compactions.boundary"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!("2", std::str::from_utf8(&raw_boundary).unwrap());

        assert!(compactions_store
            .try_read_compactions(1)
            .await
            .unwrap()
            .is_some());
        assert!(compactions_store
            .try_read_compactions(2)
            .await
            .unwrap()
            .is_some());
    }
}
